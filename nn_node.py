from typing import List, Dict, Optional

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
    def __init__(self, n_inputs, is_root: bool = False):
        super(BlackBoxNode, self).__init__()
        self.flatten: nn.Flatten = nn.Flatten()
        self.black_box: nn.Sequential = create_black_box(n_inputs)
        self.is_root = is_root

    def forward(self, *inputs) -> torch.Tensor:
        inputs = torch.cat(inputs, 0)
        x = self.flatten(inputs)
        out = self.black_box(x)
        if self.is_root:
            out = out.squeeze()
        elif out.dim() == 2:
            out = out.unsqueeze(1)
        return out


class GreyBoxNode(NnNode):
    def __init__(self, operation, child_nodes: List[NnNode], child_input_idxs=None, is_root: bool = False):
        super(GreyBoxNode, self).__init__()
        self.operation: Operation = operation
        self.child_nodes = nn.ModuleList(child_nodes)
        self.child_input_idxs: Optional[Dict[NnNode, List[int]]] = child_input_idxs
        self.is_root = is_root

    def forward(self, *inputs) -> torch.Tensor:
        if len(inputs) != 0:
            if inputs[0].dim() == 2:
                inputs = torch.cat(inputs, 0).unsqueeze(1)
            else:
                inputs = torch.cat(inputs, 0)
        if len(self.child_nodes) == 1:
            child_output: torch.Tensor = self.child_nodes[0](inputs)
            out = self.operation(child_output)
            return out

        children_outputs = []
        for child in self.child_nodes:
            child_inputs = inputs[:,:,self.child_input_idxs[child]]
            children_outputs.append(child(child_inputs))
        children_outputs = torch.cat(children_outputs, 2)
        out = self.operation(children_outputs)
        if self.is_root:
            return out.squeeze()
        return out


class LeafNode(NnNode):
    def __init__(self, add_linear_layer: bool = True):
        super(LeafNode, self).__init__()
        self.add_linear_layer = add_linear_layer
        if add_linear_layer:
            self.linear_layer = nn.Linear(1, 1)

    def forward(self, *inputs) -> torch.Tensor:
        if len(inputs) != 0:
            if inputs[0].dim() == 2:
                inputs = torch.cat(inputs, 0).unsqueeze(1)
            else:
                inputs = torch.cat(inputs, 0)

        if inputs.size(2) != 1:
            raise TypeError("input to leaf node has to be a singular variable, received ", inputs)
        if self.add_linear_layer:
            inputs = self.linear_layer(inputs)
        return inputs
