from typing import List, Dict

import torch
import abc
from abc import abstractmethod
from torch import nn
from black_box import create_black_box
from operations import Operation


class NnNode(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(NnNode, self).__init__()

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass


class BlackBoxNode(NnNode):
    def __init__(self, n_inputs):
        super(BlackBoxNode, self).__init__()
        self.flatten: nn.Flatten = nn.Flatten()
        self.black_box: nn.Sequential = create_black_box(n_inputs)

    def forward(self, *inputs) -> torch.Tensor:
        inputs = torch.cat(inputs, 1)
        x = self.flatten(inputs)
        out = self.black_box(x)
        return out


class GreyBoxNode(NnNode):
    def __init__(self, operation, child_nodes, child_input_idxs=None):
        super(GreyBoxNode, self).__init__()
        self.operation: Operation = operation
        self.child_nodes: List[NnNode] = child_nodes
        self.child_input_idxs: Dict[NnNode, List[int]] = child_input_idxs

    def forward(self, *inputs) -> torch.Tensor:
        if len(self.child_nodes) == 1:
            child_output: torch.Tensor = self.child_nodes[0](torch.cat(inputs, 1))
            out = self.operation(child_output)
            return out

        children_outputs = []
        for child in self.child_nodes:
            child_inputs = torch.cat([inputs[idx] for idx in self.child_input_idxs[child]], 1)
            children_outputs.append(child(child_inputs))
        out = self.operation(children_outputs)
        return out
