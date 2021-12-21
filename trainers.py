import torch
from torch import nn
from torch.utils.data import DataLoader

from nn_node import NnNode


class ModelTrainer:
    def __init__(self,
                 model: NnNode,
                 epochs: int,
                 lr: float,
                 max_lr: float,
                 train_loader: DataLoader,
                 show_losses: bool = False):
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

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            print(epoch + 1, "/", self.epochs)
            for batch, (x, y) in enumerate(self.train_loader):
                x, y = (x.to(self.device), y.to(self.device))
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.show_losses and batch % 50 == 0:
                    print(loss.item())


class MetaTrainer:

    @staticmethod
    def test_additive_separability(model, distribution: torch.distributions.Distribution, n_tests=20):
        x = distribution.sample((n_tests,)).requires_grad_(True)
        y = model(x)
        dydx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        d2ydx = torch.stack([torch.autograd.grad(dydx[:, i].sum(), x, create_graph=True)[0] for i in range(3)], dim=2)
        print(d2ydx)
        # TODO: finish implementing the test properly

    @staticmethod
    def test_multiplicative_separability(model, distribution: torch.distributions.Distribution, n_tests=20):
        x = distribution.sample((n_tests,)).requires_grad_(True)
        y = model(x)
        dydx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        d2ydx_divf = torch.stack(
            [torch.autograd.grad((dydx[:, i] / y).sum(), x, create_graph=True)[0] for i in range(3)], dim=2)
        print(d2ydx_divf)
        # TODO: finish implementing the test properly

    def __init__(self):
        # TODO: implement this properly

        pass

    # TODO: implement the training part

    pass
