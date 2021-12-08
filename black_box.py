import torch
from torch import nn


def create_black_box(n_inputs: int) -> nn.Sequential:
    black_box = nn.Sequential(
        nn.Linear(n_inputs, n_inputs * 2 // 3),
        nn.ReLU(),
        nn.Linear(n_inputs * 2 // 3, n_inputs * 2 // 3),
        nn.ReLU(),
        nn.Linear(n_inputs * 2 // 3, 1),
    )
    return black_box
