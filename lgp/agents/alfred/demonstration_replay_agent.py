from typing import Dict
import torch
import random
import itertools

from lgp.abcd.agent import Agent
from lgp.abcd.repr.state_repr import StateRepr

from lgp.env.teach.teach_observation import TeachObservation
from lgp.env.teach.tasks import TeachTask
from lgp.env.teach.teach_action import TeachAction, ACTION_TYPES

from lgp.models.alfred.handcoded_skills.init_skill import InitSkill

import lgp.env.blockworld.config as config


class DemonstrationReplayAgent(Agent):
    def __init__(self):
        super().__init__()
        self.actions = None
        self.init_skill = InitSkill()
        self.initialized = False
        self.current_step = 0

    def get_trace(self, device="cpu") -> Dict:
        return {}

    def clear_trace(self):
        ...

    def start_new_rollout(self, task: TeachTask, state_repr: StateRepr = None):
        self.actions = task.traj_data.get_api_action_sequence()
        self.current_step = 0
        self.initialized = False
        self.init_skill.start_new_rollout()

    def act(self, observation: TeachObservation) -> TeachAction:
        # First run the init skill until it stops
        if not self.initialized:
            action = self.init_skill.act(...)
            if action.is_stop():
                self.initialized = True
            else:
                return action

        # Then execute the prerecorded action sequence
        if self.current_step < len(self.actions):
            action = self.actions[self.current_step]
        else:
            action = TeachAction(action_type="Stop", obj_coord=(None,None), oid=None, action={"action_id":0, "action_idx":0, "action_name": "Stop"})

        self.current_step += 1
        return action
