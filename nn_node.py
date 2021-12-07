from typing import List, Dict

import torch
import abc
from abc import abstractmethod
from torch import nn
from black_box import create_black_box


class NnNode(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(NnNode, self).__init__()

    @abstractmethod
    def forward(self, inputs):
        pass


class BlackBoxNode(NnNode):
    def __init__(self, n_inputs):
        super(BlackBoxNode, self).__init__()
        self.flatten = nn.Flatten()
        self.black_box = create_black_box(n_inputs)

    def forward(self, *inputs):
        x = self.flatten(inputs)
        out = self.black_box(x)
        return out


class GreyBoxNode(NnNode):
    def __init__(self, operation, children, child_input_idxs=None):
        super(GreyBoxNode, self).__init__()
        self.operation = operation
        self.children: List[NnNode] = children  # operation no of inputs must match
        self.child_input_idxs: Dict[NnNode, List[int]] = child_input_idxs

    def forward(self, *inputs):
        if len(self.children) == 1:
            child_out = self.children[0].forward(inputs)
            out = self.operation(child_out)
            return out

        children_outputs = dict()
        for child in self.children:
            child_inputs = [inputs[idx] for idx in self.child_input_idxs[child]]
            children_outputs[child] = child.forward(child_inputs)
        out = self.operation(children_outputs)
        return out
