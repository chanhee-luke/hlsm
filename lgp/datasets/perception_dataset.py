from abc import abstractmethod

from typing import List, Dict, Union

from lgp.abcd.dataset import ExtensibleDataset

from lgp.env.teach.teach_observation import TeachObservation
from lgp.rollout.rollout_data import load_rollout_from_path

from lgp.env.teach.teach_subgoal import IDX_TO_ACTION_IDX

import torch


class PerceptionDataset(ExtensibleDataset):

    def __init__(self, chunk_paths):
        self.chunk_paths = chunk_paths

    @abstractmethod
    def __getitem__(self, item):
        example = load_rollout_from_path(self.chunk_paths[item])
        example = self._process_example(example)
        return example

    @abstractmethod
    def __len__(self):
        return len(self.chunk_paths)

    def _process_example(self, example):
        # tmp mapping
        example["subgoal"].action_type = torch.tensor([IDX_TO_ACTION_IDX[int(example["subgoal"].action_type.item())]]) #FIXME tmp mapping

        observation = example["observation"]

        observation.data_augment()

        example_out = {
            "observation": observation
        }
        return example_out

    # Inherited from lgp.abcd.dataset.ExtensibleDataset
    def collate_fn(self, list_of_examples: Union[List[Dict], List[List[Dict]]]) -> Dict:
        list_of_examples = [l for l in list_of_examples if l is not None]

        observations = [l["observation"] for l in list_of_examples]
        observations = TeachObservation.collate(observations)

        out = {
            "observations": observations
        }
        return out
