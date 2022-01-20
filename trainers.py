from collections import deque
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

import black_box
import generated_dataset
import nn_node
import operations
import utils
from nn_node import NnNode


class ModelTrainer:
    def __init__(self,
                 model: NnNode,
                 epochs: int,
                 lr: float,
                 max_lr: float,
                 train_loader: DataLoader,
                 show_losses: bool = False,
                 add_additive_separability_loss: bool = False,
                 distribution=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.train_loader = train_loader
        self.show_losses = show_losses
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader)
        )
        self.add_additive_separability_loss = add_additive_separability_loss
        if add_additive_separability_loss:
            self.distribution = distribution
        self.model_loss = None

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            print(epoch + 1, "/", self.epochs)
            for batch, (x, y) in enumerate(self.train_loader):
                x, y = (x.to(self.device), y.to(self.device))
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                if self.add_additive_separability_loss:
                    loss += torch.sum(torch.abs())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.show_losses and batch % 50 == 0:
                    print(loss.item())
                self.model_loss = loss

    def get_loss(self):
        return self.model_loss


class MetaTrainer:

    @staticmethod
    def get_hessian(node, distribution, device, n_tests=30, divide_by_f=False):
        x = distribution.sample((n_tests,)).requires_grad_(True)[:, node.input_set]
        n = x.size()[1]
        x = x.to(device)
        y = node(x)
        dydx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        if divide_by_f:
            d2ydx_divf = torch.stack(
                [torch.autograd.grad((dydx[:, i] / y).sum(), x, create_graph=True)[0] for i in range(n)], dim=2)
            return torch.sum(d2ydx_divf, axis=0) / n_tests
        else:
            d2ydx = torch.stack([torch.autograd.grad(dydx[:, i].sum(), x, create_graph=True)[0] for i in range(n)], dim=2)
            return torch.sum(d2ydx, axis=0) / n_tests

    @staticmethod
    def separate_variables_by_component(derivative_matrix: torch.tensor):
        n = len(derivative_matrix)
        unmarked_vars: List[bool] = [True for _ in range(n)]
        component_map = []
        component_idx = 0
        variable_queue = deque()
        for i in range(n):
            if unmarked_vars[i]:
                variable_queue.append(i)
                new_component = []
                # main dfs loop to mark out components

                while variable_queue:
                    current_var = variable_queue.popleft()
                    if unmarked_vars[current_var]:
                        unmarked_vars[current_var] = False
                        new_component.append(current_var)
                        neighbours = torch.arange(n, dtype=torch.int32)[derivative_matrix[current_var]].tolist()
                        for neighbour in neighbours:
                            variable_queue.append(neighbour)
                component_idx += 1
                component_map.append(new_component)
        return component_map

    @staticmethod
    def separate_variables_by_clique(derivative_matrix: torch.tensor):
        pass

    def __init__(self, train_dataset, n_inputs, distribution, model=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_inputs = n_inputs
        self.distribution = distribution
        self.model = black_box.create_black_box(n_inputs) if model is None else model
        self.dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.model_loss = None

    def training_step(self, new_model):
        trainer = ModelTrainer(
            model=new_model,
            epochs=40,
            lr=0.001,
            max_lr=0.005,
            train_loader=self.dataloader,
            show_losses=True,
            add_additive_separability_loss=True,
            distribution=self.distribution,
        )
        trainer.train()
        new_loss = trainer.get_loss()
        return new_loss

    def train(self, steps=None):
        self.model_loss = self.training_step(self.model)
        if steps is None:
            pass
        else:
            for i in range(steps):
                black_box_nodes = self.model.get_black_box_nodes()
                for black_box_node in black_box_nodes:
                    possible_split = self.test_additive_separability(black_box_node)
                    if len(possible_split) == 1:
                        possible_split = self.test_additive_separability(black_box_node)
                        if len(possible_split) == 1:
                            # TODO: apply monovariate operations to reduce loss
                            pass
                        else:
                            # TODO: apply multiplicative split to reduce loss
                            pass
                    else:
                        new_leaves = [nn_node.BlackBoxNode(len(input_set), input_set) for input_set in possible_split]
                        leaf_inputs = {leaf: [leaf.input_set] for leaf in new_leaves}
                        new_node = nn_node.GreyBoxNode(
                            operation=operations.MultivariateOperation(operations.MultivariateOp.ADD),
                            child_nodes=new_leaves,
                            child_input_idxs=leaf_inputs,
                            is_root=black_box_node.is_root,
                            input_set=black_box_node.input_set
                        )
                        # TODO: replace former node with new one
                        new_model = NnNode()

                    new_loss = self.training_step(new_model)
                    if new_loss < self.model_loss:
                        self.model = new_model
                        self.model_loss = new_loss
                        break

    def test_additive_separability(self, node, cutoff=.5, n_tests=100, check_cliques=False):
        d2ydx = MetaTrainer.get_hessian(node, self.distribution, self.device, n_tests)
        non_zero = abs(d2ydx > cutoff)
        non_zero = non_zero & non_zero.T
        return MetaTrainer.separate_variables_by_component(non_zero)

    def test_multiplicative_separability(self, node, cutoff=.5, n_tests=100, check_cliques=False):
        d2ydx_divf = MetaTrainer.get_hessian(node, self.distribution, self.device, n_tests, divide_by_f=True)
        non_zero = abs(d2ydx_divf > cutoff)
        non_zero = non_zero & non_zero.T
        return MetaTrainer.separate_variables_by_component(non_zero)


if __name__ == '__main__':
    t = torch.tensor([[True, False, False], [True, False, True], [False, True, False]])
    print(MetaTrainer.separate_variables_by_component(t & t.T))
