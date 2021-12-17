import torch
import nn_node
from datetime import datetime


def save_model(model: torch.nn.Module, name: str):
    path = 'res/models/'

    torch.save(model.state_dict(), path + name + datetime.now().isoformat() + '.pt')


def load_model(model: torch.nn.Module, model_name: str):
    path = 'res/models/' + model_name + '.pt'
    model.load_state_dict(torch.load(path))
