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
    def __init__(self, operation, children, child_input_idxs=None):
        super(GreyBoxNode, self).__init__()
        self.operation: Operation = operation
        self.children: List[NnNode] = children  # operation no of inputs must match no of children
        self.child_input_idxs: Dict[NnNode, List[int]] = child_input_idxs

    def forward(self, *inputs) -> torch.Tensor:
        if len(self.children) == 1:
            child_output: torch.Tensor = self.children[0].forward(torch.cat(inputs, 1))
            out = self.operation.forward(child_output)
            return out

        children_outputs = []
        for child in self.children:
            child_inputs = torch.cat([inputs[idx] for idx in self.child_input_idxs[child]],1)
            children_outputs.append(child.forward(child_inputs))
        out = self.operation.forward(children_outputs)
        return out
