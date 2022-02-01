from typing import List, Dict, Optional

import torch
import abc
from abc import abstractmethod
from torch import nn

import operations
from black_box import create_black_box
from operations import Operation


class NnNode(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(NnNode, self).__init__()

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_black_box_nodes(self) -> List:
        pass

    @abstractmethod
    def set_parent(self, parent):
        pass

    @abstractmethod
    def get_parent(self):
        pass


class BlackBoxNode(NnNode):
    def __init__(self, n_inputs, input_set: List[int] = None, parent: Optional[NnNode] = None):
        super(BlackBoxNode, self).__init__()
        self.flatten: nn.Flatten = nn.Flatten()
        self.black_box: nn.Sequential = create_black_box(n_inputs)
        self.parent = None if parent is None else [parent]  # stored in a list to prevent registering module parameters
        self.is_root = parent is None
        if input_set is None:
            self.input_set = list(range(n_inputs))
        else:
            self.input_set = input_set

    def forward(self, *inputs) -> torch.Tensor:
        inputs = torch.cat(inputs, 0)
        x = self.flatten(inputs)
        out = self.black_box(x)
        if self.is_root:
            out = out.squeeze()
        elif out.dim() == 2:
            out = out.unsqueeze(1)
        return out

    def get_black_box_nodes(self) -> List[NnNode]:
        return [self]

    def set_parent(self, parent: NnNode):
        self.parent = None if parent is None else [parent]
        self.is_root = parent is None

    def get_parent(self):
        return None if self.parent is None else self.parent[0]


class GreyBoxNode(NnNode):
    def __init__(self,
                 operation,
                 input_set: List[int],
                 child_nodes: List[NnNode],
                 child_input_idxs=None,
                 parent: NnNode = None):
        super(GreyBoxNode, self).__init__()
        self.operation: Operation = operation
        self.child_nodes = nn.ModuleList(child_nodes)
        for child_node in child_nodes:
            child_node.set_parent(self)  # same as always
        self.child_input_idxs: Optional[Dict[NnNode, List[int]]] = child_input_idxs
        self.parent = None if parent is None else [parent]  # stored in a list to prevent registering module parameters
        self.is_root = parent is None
        self.input_set = input_set

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
            child_inputs = inputs[:, :, self.child_input_idxs[child]]
            children_outputs.append(child(child_inputs))
        children_outputs = torch.cat(children_outputs, 2)
        out = self.operation(children_outputs)
        if self.is_root:
            return out.squeeze()
        return out

    def get_black_box_nodes(self) -> List[NnNode]:
        return sum([child.get_black_box_nodes() for child in self.child_nodes], [])

    def set_parent(self, parent: NnNode):
        self.parent = None if parent is None else [parent]
        self.is_root = parent is None

    def get_parent(self):
        return None if self.parent is None else self.parent[0]


class LeafNode(NnNode):
    def __init__(self, input_idx: int, add_linear_layer: bool = True, parent: NnNode = None):
        super(LeafNode, self).__init__()
        self.add_linear_layer = add_linear_layer
        if add_linear_layer:
            self.linear_layer = nn.Linear(1, 1)
        self.input_set = [input_idx]
        self.parent = None if parent is None else [parent]

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

    def get_black_box_nodes(self) -> List[NnNode]:
        return []

    def set_parent(self, parent: NnNode):
        self.parent = None if parent is None else [parent]

    def get_parent(self):
        return None if self.parent is None else self.parent[0]


if __name__ == '__main__':
    bp = BlackBoxNode(3)
    bc = BlackBoxNode(3, parent=bp)
    lp = GreyBoxNode(operations.UnivariateOperation(operations.UnivariateOp.LOG),
                     input_set=[0, 1, 2],
                     child_nodes=[bc]
                     )
    print(bc, bc.parent)