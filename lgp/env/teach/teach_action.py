from typing import Union
from lgp.abcd.action import Action

import numpy as np
import torch, sys, pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

IDX_TO_ACTION_TYPE = {
    0: 'Stop',
    2: 'Forward',
    3: 'Backward',
    4: 'Turn Left',
    5: 'Turn Right',
    6: 'Look Up',
    7: 'Look Down',
    8: 'Pan Left',
    9: 'Pan Right',
    200: 'Pickup',
    201: 'Place',
    202: 'Open',
    203: 'Close',
    204: 'ToggleOn',
    205: 'ToggleOff',
    206: 'Slice',
    211: 'Pour'
}

# TODO: Reinstate Stop action as an action type

ACTION_TYPE_TO_IDX = {v:k for k,v in IDX_TO_ACTION_TYPE.items()}
ACTION_TYPES = list(IDX_TO_ACTION_TYPE.values())#[IDX_TO_ACTION_TYPE[i] for i in range(len(IDX_TO_ACTION_TYPE))] #TODO why this?

NAV_ACTION_TYPES = [
    'Forward',
    'Backward',
    'Turn Left',
    'Turn Right',
    'Look Up',
    'Look Down',
    'Pan Left',
    'Pan Right'
]

INTERACT_ACTION_TYPES = [
    "Pickup",
    "Place",
    "Open",
    "Close",
    "ToggleOn",
    "ToggleOff",
    "Slice",
    "Pour"
]

class TeachAction(Action):
    def __init__(self,
                 action_type: str,
                 action,
                 obj_coord=(None, None),
                 oid=""):
        super().__init__()
        self.action_type = action_type
        self.obj_coord = obj_coord
        self.oid = oid
        self.api_action = action

    # def to(self, device):
    #     self.argument_mask = self.argument_mask.to(device) if self.argument_mask is not None else None
    #     return self

    @classmethod
    def stop_action(cls):
        return cls("Stop", cls.get_empty_obj_coord())

    @classmethod
    def get_empty_obj_coord(cls) -> torch.tensor:
        return (None, None)

    @classmethod
    def get_action_type_space_dim(cls) -> int:
        return len(ACTION_TYPE_TO_IDX)

    @classmethod
    def action_type_str_to_intid(cls, action_type_str : str) -> int:
        return ACTION_TYPE_TO_IDX[action_type_str]

    @classmethod
    def action_type_intid_to_str(cls, action_type_intid : int) -> str:
        return IDX_TO_ACTION_TYPE[action_type_intid]

    @classmethod
    def get_interact_action_list(cls):
        return INTERACT_ACTION_TYPES

    @classmethod
    def get_nav_action_list(cls):
        return NAV_ACTION_TYPES

    def is_valid(self):
        if self.action_type in NAV_ACTION_TYPES:
            return True
        elif self.obj_coord is None:
            print("TeachAction::is_valid -> missing argument mask")
            return False
        elif self.obj_coord.sum() < 1:
            print("TeachAction::is_valid -> empty argument mask")
            return False
        return True

    def type_intid(self):
        return self.action_type_str_to_intid(self.action_type)

    def type_str(self):
        return self.action_type

    # 
    # def to_teach_api(self) -> (str, Union[None, np.ndarray]):
    #     if self.action_type in NAV_ACTION_TYPES:
    #         argmask_np = None
    #     else: # Interaction action needs a mask
    #         if self.argument_mask is not None:
    #             if isinstance(self.argument_mask, torch.Tensor):
    #                 argmask_np = self.argument_mask.detach().cpu().numpy()
    #             else:
    #                 argmask_np = self.argument_mask
    #         else:
    #             argmask_np = None
    #     return self.action_type, argmask_np

    def is_stop(self):
        return self.action_type == "Stop"

    def __eq__(self, other: "TeachAction"):
        return self.action_type == other.action_type and self.obj_coord == other.obj_coord

    def __str__(self):
        return self.action_type

    def represent_as_image(self):
        #TODO 
        raise NotImplementedError
