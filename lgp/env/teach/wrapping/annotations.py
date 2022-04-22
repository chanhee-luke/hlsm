"""
This file includes code to index, load, and manage the traj_data json files from TEACH.
NOTE Modified to take in TEACh EDH instances and game files at the same time!
"""
from typing import List, Dict, Union
import os, pdb
import json
import copy

from lgp.env.teach.teach_action import TeachAction
from lgp.env.teach.wrapping.paths import get_teach_root_path, get_splits_path, get_task_dir_path, get_traj_data_paths, get_task_traj_data_path, get_game_dir_path


class TrajData:
    def __init__(self, traj_data_path, game_dir_path):
        # Open edh instance
        try:
            with open(traj_data_path, "r") as fp:
                data = json.load(fp)
        except json.decoder.JSONDecodeError as e:
            print(f"Couldn't load json: {traj_data_path}")
            raise e

        # Get game file id from edh instance file
        game_id = data["game_id"]
        game_data_path = os.path.join(game_dir_path, f"{game_id}.game.json")

        # Open game data file
        try:
            with open(game_data_path, "r") as fp:
                game_data = json.load(fp)
        except json.decoder.JSONDecodeError as e:
            print(f"Couldn't load json: {game_data_path}")
            raise e
        self.edh = data
        self.game_data = game_data

    def is_test(self):
        # Lazy way to detect test examples - they don't have ground-truth plan data
        #NOTE Not used in EDH since test data is unknown
        return "plan" not in self.edh

    def patch_trajectory(self):
        # The ground-truth trajectories are generated with a PDDL planner that has access to ground truth state.
        # as such, it is unconcerned about observability and exploration, and often walks around with it's head
        # down not seeing anything. This is terrible for building voxel map representations of the world.
        # Here we patch the ground truth sequences by making sure that the agent always walks around with it's head
        # tilted down 30 degrees, and then tilts to the correct angle before taking each action.
        # Sometimes this results in invalid action sequences, when the object the agent is holding collides with
        # the environment.

        #NOTE This has been modified to match EDH instances
        self.fix_lookdown()
        #self.add_rotate_explore()

    """
    def add_rotate_explore(self):
        proto_rotateleft = {
            "api_action": {
                "action": "RotateLeft",
                "forceAction": True
            },
            "discrete_action": {
                "action": "RotateLeft_90",
                "args": {}
            },
            "high_idx": 0
        }
        self.edh["plan"]["low_actions"] = [proto_rotateleft for _ in range(4)] + self.edh["plan"]["low_actions"]
    """

    def fix_lookdown(self):
        old_plan = self.edh["driver_actions_future"]
        plan = copy.deepcopy(old_plan)
        n = len(plan)

        proto_ld = {
            "action_id": 7,
            "action_idx": 7,
            "obj_interaction_action": 0,
            "action_name": "Look Down",
            "time_start": -1,
            "oid": null,
            "x": null,
            "y": null
        }

        proto_lu = {
            "action_id": 6,
            "action_idx": 6,
            "obj_interaction_action": 0,
            "action_name": "Look Up",
            "time_start": -1,
            "oid": null,
            "x": null,
            "y": null
        }

        # First mark for each action (except LookUp, LookDown), how many lookdowns have been done
        step_ldc = []
        ldc = 0
        for i in range(n):
            act_i = plan[i]["action_name"]
            if act_i == "Look Down":
                ldc += 1
            elif act_i == "Look Up":
                ldc -= 1
            else:
                step_ldc.append(ldc)

        # Then delete all LookDown and LookUp actions
        for i in range(n-1, -1, -1):
            act_i = plan[i]["action_name"]
            if act_i in ["Look Down", "Look Up"]:
                plan = plan[:i] + plan[i+1:]

        assert len(plan) == len(step_ldc)

        # Then insert the right amount of LookDown and LookUp around interaction actions
        new_plan = []
        for i in range(len(step_ldc)):
            act_i = plan[i]["action_name"]
            if act_i in ["Pickup", "Place", "Open", "Close", "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                ld = copy.deepcopy(proto_ld)
                lu = copy.deepcopy(proto_lu)
                for c in range(step_ldc[i]):
                    new_plan.append(ld)
                for c in range(0, -step_ldc[i], 1):
                    new_plan.append(lu)
                new_plan.append(plan[i])
                for c in range(step_ldc[i]):
                    new_plan.append(lu)
                for c in range(0, -step_ldc[i], 1):
                    new_plan.append(ld)
            else:
                new_plan.append(plan[i])

        # Replace the plan
        self.edh["driver_actions_future"] = new_plan

    def iterate_strings(self):
        # Iterate through task descriptions
        task_desc = self.get_task_description()
        yield task_desc
        step_descs = self.get_step_descriptions()
        for step_desc in step_descs:
            yield step_desc

    def get_task_id(self):
        #NOTE EDH instance id, unique to each EDH instance
        return self.edh['instance_id']

    def get_task_type(self):
        #NOTE EDH instance task description
        return self.game_data['tasks'][0]["task_name"]

    def get_task_description(self):
        #NOTE EDH does not have high level description so task description == step description
        #print("***USING TASK DESCRIPTION, ARE YOU SURE???***")
        dialogues =  ' '.join(self.edh["dialog_history"][0])
        print(dialogues)
        return dialogues
        #return self.game_data['tasks'][0]['desc']

    def get_step_descriptions(self):
        # Dialogue history (Originally low level instruction in ALFRED) 
        print("***USING DIALOGUE HISTORY!!!***")
        return self.edh["dialog_history"]

    def get_world_number(self):
        # Floorplan number (to initialize simulator)
        return self.game_data['tasks'][0]['episodes'][0]["world"]
    
    def get_world_type_number(self):
        # Floorplan number (to initialize simulator)
        return self.game_data['tasks'][0]['episodes'][0]["world_type"]

    def get_object_poses(self):
        object_poses = []
        for obj in self.game_data['tasks'][0]['episodes'][0]['initial_state']['objects']:
            object_poses.append({"objectName": obj["name"], \
                "position": obj["position"], \
                "rotation": obj["rotation"],
            })
        return object_poses

    def get_initial_state(self):
        return self.game_data['tasks'][0]['episodes'][0]['initial_state']

    def get_dirty_and_empty(self):
        # Checks if any objects are dirty and empty
        dirty_and_empty = False
        for i in range(len(self.game_data['tasks'][0]['episodes'][0]['initial_state']['objects'])):
            if self.game_data['tasks'][0]['episodes'][0]['initial_state']['objects'][i]["isDirty"]:
                dirty_and_empty = True
                break
        return dirty_and_empty

    def get_object_toggles(self):
        # Getting toggled object states
        toggled_obj_data = []
        objects = self.game_data['tasks'][0]['episodes'][0]['initial_state']['objects']
        for obj in objects:
            toggled_obj_data.append({'isOn': obj['isToggled'], 'objectType': obj['objectType']})
        return toggled_obj_data

    def get_init_actions(self) -> List:
        # In EDH, init_action is a list of actions to get to that location
        return self.edh["driver_action_history"]

    def get_low_actions(self) -> List:
        return self.edh["driver_actions_future"]

    def get_init_action_sequence(self) -> Union[List[Dict], None]:
        sequence = self.get_init_actions()
        init_action_sequence = [TeachAction(
            action_type=a["action_name"],
            obj_coord=(a["x"], a["y"]),
            oid=a["oid"],
            action=a         
         ) for a in sequence]
        return init_action_sequence

    def get_api_action_sequence(self) -> Union[List[Dict], None]:
        sequence = self.get_low_actions()
        api_ish_sequence = [TeachAction(
            action_type=a["action_name"],
            obj_coord=(a["x"], a["y"]),
            oid=a["oid"],
            action=a        
        ) for a in sequence]
        return api_ish_sequence


class TeachAnnotations():

    def __init__(self):
        self.splits = get_splits_path()
        # with open(self.splits_path, "r") as fp:
        #     splits = json.load(fp)
        # self.splits = sesplits_path

    @classmethod
    def load_traj_data_for_task(cls, data_split: str, task_id: str) -> TrajData:
        traj_data_path = get_task_traj_data_path(data_split, task_id)
        game_dir_path = get_game_dir_path(data_split)
        traj_data = TrajData(traj_data_path, game_dir_path)
        return traj_data

    @classmethod
    #TODO if this is used, modify
    def load_all_traj_data(cls, data_split: str) -> List[TrajData]:
        traj_data_paths = get_traj_data_paths(data_split)
        traj_datas = [TrajData(t) for t in traj_data_paths]
        return traj_datas

    def get_teach_data_splits(self) -> (List[str], Dict[str, List]):
        """
        Return:
            (list, dict)
            list - list of strings of datasplit names
            dict - dictionary, indexed by datasplit names, containing list of dictionaries of format:
                {"repeat_idx": int, "task": str}
        """
        list_of_splits = list(sorted(self.splits.keys()))
        return list_of_splits, self.splits

    def get_all_task_ids_in_split(self, datasplit: str = "train") -> List[str]:
        assert datasplit in self.splits, f"Datasplit {datasplit} not found in available splits: {self.splits.keys()}"
        task_ids = list(sorted(set([d["task"] for d in self.splits[datasplit]])))
        return task_ids

