import torch
from torch import nn


def create_mlp(n_inputs: int) -> nn.Sequential:
    layer_size = max(8, n_inputs * 2 // 3)
    black_box = nn.Sequential(
        nn.Linear(n_inputs, layer_size),
        nn.Tanh(),
        nn.Linear(layer_size, layer_size),
        nn.Tanh(),
        nn.Linear(layer_size, 1),
    )
    return black_box
