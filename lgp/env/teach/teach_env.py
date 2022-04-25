import sys
import os
import copy
from typing import Tuple, Dict, Iterator, Union, Collection

from lgp.abcd.env import Env

from lgp.simulators import simulator_factory

from lgp.env.teach.state_tracker import StateTracker
from lgp.env.teach.wrapping.args import get_faux_args

from lgp.env.teach.tasks import TeachTask, TaskRecord
from lgp.env.teach.teach_observation import TeachObservation
from lgp.env.teach.teach_action import TeachAction, INTERACT_ACTION_TYPES

from lgp.env.teach import config
from lgp.utils.utils import SimpleProfiler
from lgp.settings import get_settings
from lgp.logger import create_logger
from lgp.utils.utils import reduce_float_precision

logger = create_logger(__name__)

PROFILE = False

DEFAULT_SETUP = {
    "data_splits": ["train"],
    "filter_task_types": [],
    "no_segmentation": False,
    "no_depth": False,
    "max_fails": 10
}

import sys, pdb


class TeachEnv(Env):

    def __init__(self, device=None, setup=None, hparams=None):
        super().__init__()
        teach_display = (os.environ.get("ALFRED_DISPLAY")
                          if "ALFRED_DISPLAY" in os.environ
                          else os.environ.get("DISPLAY"))
        if teach_display.startswith(":"):
            teach_display = teach_display[1:]
        #self.thor_env = TEAChController(x_display=teach_display, base_dir=get_settings().AI2THOR_BASE_DIR) #FIXME what is this?
        self.simulator = simulator_factory.factory.create(simulator_name="thor", web_window_size=300)  #NOTE TEACH actually uses 900, using 300 for memory issue
        self.task = None
        self.steps = 0
        self.device = device
        self.horizon : int = config.DEFAULT_HORIZON
        self.fail_count : int = 0

        if not setup:
            self.setup = DEFAULT_SETUP
        else:
            self.setup = setup

        self.data_splits = self.setup["data_splits"]
        # Optionally filter tasks by type
        allowed_tasks = self.setup["filter_task_types"]
        allowed_ids = self.setup.get("filter_task_ids", None)

        # Setup state tracker
        reference_seg = self.setup.get("reference_segmentation", False)
        reference_depth = self.setup.get("reference_depth", False)
        reference_inventory = self.setup.get("reference_inventory", False)
        reference_pose = self.setup.get("reference_pose", True)
        print(f"USING {'REFERENCE DEPTH' if reference_depth else 'PREDICTED DEPTH'} "
              f"and {'REFERENCE SEGMENTATION' if reference_seg else 'PREDICTED SEGMENTATION'}")

        self.max_fails = setup.get("max_fails", 10)
        print(f"Max failures: {self.max_fails}")
        self.state_tracker = StateTracker(reference_seg=reference_seg,
                                          reference_depth=reference_depth,
                                          reference_inventory=reference_inventory,
                                          reference_pose=reference_pose,
                                          hparams=hparams)

        if allowed_tasks is not None:
            print(f"FILTERING TASKS: {allowed_tasks}")
            task_filter = TeachTask.make_task_type_filter(allowed_tasks)
        elif allowed_ids is not None:
            print(f"FILTERING TASKS: {allowed_ids}")
            task_filter = TeachTask.make_task_id_filter(allowed_ids)
        else:
            raise ValueError("")
        self.task_iterator = TeachTask.iterate_all_tasks(data_splits=self.data_splits, task_filter=task_filter)
        self.reward_type = self.setup.get("reward_type", "sparse")
        self.smooth_nav = self.setup.get("smooth_nav", False)
        self.task_num_range = None

        self.prof = SimpleProfiler(print=PROFILE)

        self.simulator_options = {"renderDepthImage":True, "renderInstanceSegmentation": True}

    def get_env_state(self) -> Dict:
        world = copy.deepcopy(self.world)
        task = copy.deepcopy(self.task)
        return {"world": world, "task": task, "steps": self.steps}

    def set_env_state(self, state: Dict):
        self.world = copy.deepcopy(state["world"])
        self.task = copy.deepcopy(state["task"])
        self.steps = state["steps"]

    def set_horizon(self, horizon):
        self.horizon = horizon

    def set_task_iterator(self, task_iterator: Union[Iterator, None]):
        self.task_iterator = task_iterator

    def set_task_num_range(self, task_num_range):
        # Setting this to different ranges on different processes allows parallelizing the data collection effort
        if task_num_range is not None:
            self.task_num_range = list(task_num_range)

    def _choose_task(self):
        #pdb.set_trace()
        assert self.task_iterator is not None, "Missing task iterator"
        try:
            while True:
                task, i = next(self.task_iterator)
                if self.task_num_range is None or i in self.task_num_range:
                    return task, i
        except StopIteration as e:
            print(e)
            raise StopIteration

    def reset(self, randomize=False, skip_tasks: Union[Collection[TaskRecord], None] = None) -> (TeachObservation, TeachTask):

        self.task, task_number = self._choose_task()

        # Skip tasks that are already completed
        task_id = self.task.get_task_id()
        if skip_tasks is not None:
            if self.task.get_record() in skip_tasks:
                print(f"Skipping task: {task_id}")
                return None, None, None
            else:
                print(f"Including task: {task_id}")

        self.fail_count = 0
        self.steps = 0

        # Apply patch to shift "LookDown" actions to right before interaction, and add four explore actions
        if not self.task.traj_data.is_test():
            self.task.traj_data.patch_trajectory()

        #NOTE Old way of doing things
        # object_poses = self.task.traj_data.get_object_poses()
        # dirty_and_empty = self.task.traj_data.get_dirty_and_empty()
        # object_toggles = self.task.traj_data.get_object_toggles()

        # self.prof.tick("proc")
        # # Resetting teach.env.simulator.ThorEnv
        # # see teach/models/eval/eval.py:setup_scene (line 100) for reference
        # self.simulator.reset(self.task.traj_data.get_scene_number(),
        #                           render_image=False,
        #                           render_depth_image=True,
        #                           render_class_image=False,
        #                           render_object_image=True)
        # self.simulator.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # Start new episode
        self.simulator.reset_stored_data()
        logger.info("Starting episode...")
        self.simulator.start_new_episode(
            world=self.task.traj_data.get_world_number(),
            world_type=self.task.traj_data.get_world_type_number(),
            commander_embodied=False,
            simulator_options=self.simulator_options
        )
        logger.info("... done")

        logger.info("Loading initial scene state...")
        _, s = self.simulator.load_scene_state(init_state=self.task.traj_data.get_initial_state())
        logger.info("... done")

        #NOTE Setting task in simulator is not supported in EDH
        # if task is not None:
        #     logger.info("Setting to custom task %s" % task)
        #     self.simulator.set_task(task=task)
        # else:
        #     logger.info("Setting task %s with task_params %s..." % (self.task, self.task_params))
        #     self.simulator.set_task_by_name(task_name=self.task, task_params=self.task_params)

        self.simulator.set_task(task=self.task.get_task_def())
        logger.info("... done")

        init_state = reduce_float_precision(self.simulator.get_current_state().to_dict())

        #NOTE Step multiple times b/c edh instances does not have a given start position
        for action in self.task.traj_data.get_init_action_sequence():
            if str(action) in INTERACT_ACTION_TYPES:
                y = action.obj_coord[1]
                x = action.obj_coord[0]
                step_success, _, _ = self.simulator.apply_object_interaction(action.api_action["action_name"], 1, x, y)
            else:
                step_success, _, _ = self.simulator.apply_motion(action.api_action["action_name"], 1)

        # If this is not a test example, estting the task here allows tracking results (e.g. goal-conditions)
        if not self.task.traj_data.is_test():
            # The only argument in args that ThorEnv uses is args.reward_config, which is kept to its default
            self.simulator.set_task(self.task.get_task_def())
        self.prof.tick("simulator_reset")
        print(f"Task: {str(self.task)}")
        event = self.simulator.controller.last_event
        self.state_tracker.reset(event)
        observation = self.state_tracker.get_observation()

        if self.device:
            observation = observation.to(self.device)

        return observation, self.task, task_number, event

    def _error_is_fatal(self, err):
        self.fail_count += 1
        if self.fail_count >= self.max_fails:
            print(f"EXCEEDED MAXIMUM NUMBER OF FAILURES ({self.max_fails})")
            return True
        else:
            return False

    def step(self, action: TeachAction) -> Tuple[TeachObservation, float, bool, Dict]:
        self.prof.tick("out")
        done = False
        step_success = False

        # The ALFRED API does not accept the Stop action, do nothing
        message = ""
        if action.is_stop():
            done = True
            transition_reward = 0
            api_action = None
            events = []

        # Execute all other actions in the ALFRED API
        else:
            if str(action) in INTERACT_ACTION_TYPES:
                y = action.obj_coord[1]
                x = action.obj_coord[0]
                step_success, err, _ = self.simulator.apply_object_interaction(action.api_action["action_name"], 1, x, y)
                # query = self.simulator.controller.step(
                #     action="GetObjectInFrame",
                #     x=x,
                #     y=y,
                #     checkVisible=False
                # )
                #object_id = query.metadata["actionReturn"]
                # print(dir(self.simulator))
                # object_id, obj = self.simulator._SimulatorTHOR__get_oid_at_frame_xy_with_affordance(x, y, self.simulator.controller.last_event, 0, {})
                # print()
                # print(f"For {action.api_action}, object id is {object_id} {obj}")
                # print()
            else:
                step_success, err, _ = self.simulator.apply_motion(action.api_action["action_name"], 1)
            api_action = action.api_action

            events = []

            if not step_success:
                fatal = self._error_is_fatal(err)
                print(f"ThorEnv {'fatal' if fatal else 'non-fatal'} Exec Error: {err}")
                if fatal:
                    done = True
                    api_action = None
                message = str(err)

        self.prof.tick("step")

        # Only process info if step is successful
        observation = None
        reward = -1
        md = None
        event = None
        if step_success:
            # Track state (pose and inventory) from RGB images and actions
            event = self.simulator.controller.last_event
            self.state_tracker.log_action(action)
            self.state_tracker.log_event(event)
            #self.state_tracker.log_extra_events(events)

            observation = self.state_tracker.get_observation()
            observation.privileged_info.attach_task(self.task) # TODO: See if we can get rid of this?
            if self.device:
                observation = observation.to(self.device)

            # Rewards and progress tracking metadata
            if not self.task.traj_data.is_test():
                (
                    _,
                    task_success,
                    _,
                    final_goal_conditions_total,
                    final_goal_conditions_satisfied,
                ) = self.simulator.check_episode_progress(self.task.get_task_def())

                #NOTE in EDH, goal-condition and subgoal are the same
                goal_satisfied = final_goal_conditions_satisfied
                goal_conditions_met = final_goal_conditions_satisfied
                task_success = goal_satisfied
                md = {
                    "success": task_success,
                    "goal_satisfied": goal_satisfied,
                    "goal_conditions_met": goal_conditions_met,
                    "message": message,
                }
            else:
                reward = 0
                md = {}

            # This is used to generate leaderboard replay traces:
            md["api_action"] = api_action

            self.steps += 1

            self.prof.tick("proc")
            self.prof.loop()
            self.prof.print_stats(20)
        return observation, reward, done, md, event, step_success
