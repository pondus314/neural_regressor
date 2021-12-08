import abc
from typing import List

import torch
from torch import nn


class Operation(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Operation, self).__init__()

    def forward(self, inputs) -> torch.Tensor:
        pass


class UnivariateOperation(Operation):
    def __init__(self, add_linear_layer: bool = True):
        super(UnivariateOperation, self).__init__()
        self.add_linear_layer = add_linear_layer
        if add_linear_layer:
            self.linear_layer = nn.Linear(1, 1)
        self.operation = None

    def forward(self, input) -> torch.Tensor:
        raise NotImplementedError()
        # implement the operation itself

        if self.add_linear_layer:
            output = self.linear_layer(output)

        return output


class MultivariateOperation(Operation):
    def __init__(self, add_linear_layer: bool = True):
        super(MultivariateOperation, self).__init__()

    def forward(self, *inputs) -> torch.Tensor:
        raise NotImplementedError()
        # implement the operation itself

        if self.add_linear_layer:
            output = self.linear_layer(output)

        return output
