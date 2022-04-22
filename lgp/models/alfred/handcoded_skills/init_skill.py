from typing import Dict

from lgp.abcd.skill import Skill

from lgp.env.teach.teach_action import TeachAction
from lgp.env.teach.teach_subgoal import TeachSubgoal
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr

from lgp.flags import LONG_INIT

if LONG_INIT:
    INIT_SEQUENCE = ["Look Down"] + ["Turn Left"] * 4 + ["Look Up"] * 3 + ["Turn Left"] * 4 + ["Look Down"] * 2 + ["Stop"]
else:
    INIT_SEQUENCE = ["Turn Left"] * 4 + ["Stop"]


class InitSkill(Skill):
    def __init__(self):
        super().__init__()
        self._reset()
        print(f"Init skill with sequence of length: {INIT_SEQUENCE}")

    @classmethod
    def sequence_length(cls):
        return len(INIT_SEQUENCE) - 1 # -1 because of the Stop action

    def start_new_rollout(self):
        self._reset()

    def _reset(self):
        self.count = 0
        self.trace = {}

    def get_trace(self, device="cpu") -> Dict:
        return self.trace

    def clear_trace(self):
        self.trace = {}

    def has_failed(self) -> bool:
        return False

    def set_goal(self, hl_action : TeachSubgoal):
        self._reset()

    def act(self, state_repr: AlfredSpatialStateRepr) -> TeachAction:

        if self.count >= len(INIT_SEQUENCE):
            raise ValueError("Init skill already output a Stop action! No futher calls allowed")
        action = TeachAction(action_type=INIT_SEQUENCE[self.count], action={"action_name": INIT_SEQUENCE[self.count]})
        self.count += 1
        return action