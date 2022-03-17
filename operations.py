import abc
import math
import operator
from abc import abstractmethod
from enum import Enum

import sympy
import torch
from torch import nn


class MultivariateOp(Enum):
    ADD = 1
    MULTIPLY = 2


class UnivariateOp(Enum):
    POWER = 1
    LOG = 2
    EXPONENTIAL = 3


UNIVARIATE_PARAMETERS = {
    UnivariateOp.POWER: 1,
    UnivariateOp.LOG: 0,
    UnivariateOp.EXPONENTIAL: 0,
}


class Operation(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(Operation, self).__init__()

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    @abstractmethod
    def symbolic(self, child_expressions):
        pass

    @abstractmethod
    def reset_weights(self):
        pass


class UnivariateOperation(Operation):
    def __init__(self, operation_type: UnivariateOp, add_linear_layer: bool = False):
        super(UnivariateOperation, self).__init__()
        self.add_linear_layer = add_linear_layer
        if add_linear_layer:
            self.linear_layer = nn.Linear(1, 1)
        self.operation_type = operation_type
        if UNIVARIATE_PARAMETERS[operation_type] != 0:
            params = torch.ones(UNIVARIATE_PARAMETERS[operation_type], requires_grad=True)
            self.params = nn.Parameter(params)
        else:
            self.params = None

    def forward(self, inputs) -> torch.Tensor:
        if self.operation_type == UnivariateOp.LOG:
            inputs = torch.abs(inputs)
            output = torch.log(inputs)
        elif self.operation_type == UnivariateOp.POWER:
            inputs = torch.abs(inputs)
            output = torch.pow(inputs, self.params[0])
        elif self.operation_type == UnivariateOp.EXPONENTIAL:
            output = torch.exp(inputs)
        else:
            raise TypeError("operation_type needs to be a UnivariateOp")

        if self.add_linear_layer:
            output = self.linear_layer(output)

        return output

    def symbolic(self, child_expressions):
        expression = child_expressions[0]
        if self.operation_type == UnivariateOp.POWER:
            result = sympy.Pow(expression, self.params[0].item())
        elif self.operation_type == UnivariateOp.LOG:
            result = sympy.log(expression)
        elif self.operation_type == UnivariateOp.EXPONENTIAL:
            result = sympy.exp(expression)
        if self.add_linear_layer:
            parameters = list(map(operator.methodcaller('item'), self.linear_layer.parameters()))  # python
            return parameters[0] * result + parameters[1]
        return result

    def reset_weights(self):
        if self.add_linear_layer:
            self.linear_layer.reset_parameters()
        # if self.params is not None:
        #     self.params = nn.Parameter(torch.ones_like(self.params))


class MultivariateOperation(Operation):
    def __init__(self, operation_type: MultivariateOp, add_linear_layer: bool = True):
        super(MultivariateOperation, self).__init__()
        self.add_linear_layer = add_linear_layer
        if add_linear_layer:
            self.linear_layer = nn.Linear(1, 1)
        self.operation_type = operation_type

    def forward(self, *inputs) -> torch.Tensor:
        if self.operation_type == MultivariateOp.MULTIPLY:
            output = torch.prod(torch.cat(inputs, 1), dim=2, keepdim=True)
        elif self.operation_type == MultivariateOp.ADD:
            output = torch.sum(torch.cat(inputs, 1), dim=2, keepdim=True)
        else:
            raise TypeError("operation_type needs to be a MultivariateOp")

        if self.add_linear_layer:
            output = self.linear_layer(output)

        return output

    def symbolic(self, child_expressions):
        if self.operation_type == MultivariateOp.ADD:
            result = sympy.Add(*child_expressions)
        elif self.operation_type == MultivariateOp.MULTIPLY:
            result = sympy.Mul(*child_expressions)
        if self.add_linear_layer:
            parameters = list(map(operator.methodcaller('item'), self.linear_layer.parameters()))
            return parameters[0] * result + parameters[1]
        return result

    def reset_weights(self):
        if self.add_linear_layer:
            self.linear_layer.reset_parameters()
