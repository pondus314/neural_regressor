import torch
from torch import nn


def create_black_box(n_inputs: int) -> nn.Sequential:
    black_box = nn.Sequential(
        nn.Linear(n_inputs, max(8,n_inputs * 2 // 3)),
        nn.ReLU(),
        nn.Linear(max(8, n_inputs * 2 // 3), max(8, n_inputs * 2 // 3)),
        nn.ReLU(),
        nn.Linear(max(8, n_inputs * 2 // 3), 1),
    )
    return black_box
