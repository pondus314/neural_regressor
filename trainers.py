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
    pass
