import torch
from torch import nn


def create_black_box(n_inputs: int) -> nn.Sequential:
    black_box = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    return black_box