from collections import deque
from typing import List
import copy

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
                # if self.add_additive_separability_loss:
                #     loss += torch.sum(torch.abs())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.show_losses and batch % 50 == 0:
                    print(loss.item())
                if self.model_loss is None:
                    self.model_loss = loss
                self.model_loss = min(loss, self.model_loss)

    def get_loss(self):
        if self.model_loss is None:
            self.model.eval()
            for batch, (x, y) in enumerate(self.train_loader):
                x, y = (x.to(self.device), y.to(self.device))
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                if self.model_loss is None:
                    self.model_loss = loss
                self.model_loss = min(loss, self.model_loss)
        return self.model_loss


class MetaTrainer:

    @staticmethod
    def get_hessian(node, distribution, device, n_tests=30, divide_by_f=False):
        x = distribution.sample((n_tests,)).requires_grad_(True)[:, node.input_set]
        n = x.size()[1]
        x = x.to(device)
        y = node(x).squeeze()
        dydx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        if divide_by_f:
            d2ydx_divf = torch.stack(
                [torch.autograd.grad((dydx[:, i] / y).sum(), x, create_graph=True)[0] for i in range(n)], dim=2)
            return torch.median(d2ydx_divf, axis=0)[0]
            # sometimes, extreme v
        else:
            d2ydx = torch.stack([torch.autograd.grad(dydx[:, i].sum(), x, create_graph=True)[0] for i in range(n)], dim=2)
            return torch.median(d2ydx, axis=0)[0]

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
                # main bfs loop to mark out components

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
        self.model = nn_node.BlackBoxNode(n_inputs) if model is None else model
        self.dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.model_loss = None

    def __training_step(self, new_model):
        trainer = ModelTrainer(
            model=new_model,
            epochs=40,
            lr=0.001,
            max_lr=0.005,
            train_loader=self.dataloader,
            show_losses=True,
            add_additive_separability_loss=False,
            distribution=self.distribution,
        )
        trainer.train()
        new_loss = trainer.get_loss()
        return new_loss

    def replace_in_parent(self, black_box_node, new_node):
        # TODO: test that this properly replaces the children
        if black_box_node.parent is None:
            self.model = new_node
        else:
            parent = black_box_node.get_parent()
            parents_children = parent.child_nodes
            for idx_child in range(len(parents_children)):
                if parents_children[idx_child] == black_box_node:
                    bb_inputs = parent.child_input_idxs.pop(black_box_node)
                    parents_children[idx_child] = new_node
                    parent.child_input_idxs[new_node] = bb_inputs
                    break

    def replace_by_leaf(self, black_box_node):
        leaf_node = nn_node.LeafNode(
            input_idx=black_box_node.input_set[0],
            add_linear_layer=True,
            parent=black_box_node.parent
        )

        self.replace_in_parent(black_box_node, leaf_node)

    def split_node(self, black_box_node, possible_split, split_multiplicatively):
        inputs_tensor = torch.tensor(black_box_node.input_set)
        new_leaves = [nn_node.BlackBoxNode(len(input_set), inputs_tensor[input_set].tolist()) for input_set in possible_split]
        print(*[leaf.input_set for leaf in new_leaves])
        leaf_inputs = {new_leaves[i]: [possible_split[i]] for i in range(len(possible_split))}
        new_node = nn_node.GreyBoxNode(
            operation=operations.MultivariateOperation(
                operations.MultivariateOp.MULTIPLY if split_multiplicatively
                else operations.MultivariateOp.ADD,
                add_linear_layer=False
            ),
            child_nodes=new_leaves,
            child_input_idxs=leaf_inputs,
            parent=black_box_node.parent,
            input_set=black_box_node.input_set,
        )

        self.replace_in_parent(black_box_node, new_node)

    def create_univariate_node(self, black_box_node, backup_model):
        for operation_type in operations.UnivariateOp:
            operation = operations.UnivariateOperation(
                operation_type=operation_type,
            )
            new_leaf = nn_node.BlackBoxNode(
                n_inputs=len(black_box_node.input_set),
                input_set=black_box_node.input_set,
            )
            new_node = nn_node.GreyBoxNode(
                operation=operation,
                child_nodes=[new_leaf],
                child_input_idxs={new_leaf: list(range(len(black_box_node.input_set)))},
                parent=black_box_node.parent,
                input_set=black_box_node.input_set,
            )

            self.replace_in_parent(black_box_node, new_node)

            new_loss = self.__training_step(self.model)
            if new_loss < self.model_loss:
                self.model_loss = new_loss
                print(self.model)
                return True
            else:
                self.model = backup_model
        return False

    def train(self, steps=None, skip_initial_training=False):
        if not skip_initial_training:
            self.model_loss = self.__training_step(self.model)
        else:
            trainer = ModelTrainer(
                model=self.model,
                epochs=1,
                lr=0.001,
                max_lr=0.005,
                train_loader=self.dataloader,
                show_losses=True,
                add_additive_separability_loss=False,
                distribution=self.distribution,
            )
            loss = trainer.get_loss()
            self.model_loss = loss

        if steps is None:
            pass
        else:
            failed_attempts = 0
            for step in range(steps):
                black_box_nodes = self.model.get_black_box_nodes()
                for black_box_node in black_box_nodes:
                    backup_model = copy.deepcopy(self.model)
                    if len(black_box_node.input_set) == 1:
                        # attempt to create univariate transformation middle node
                        result = self.create_univariate_node(black_box_node, backup_model)
                        if result:
                            break

                        # if no univariate transformation reduces loss, try to turn black box into leaf

                        self.replace_by_leaf(black_box_node)

                        new_loss = self.__training_step(self.model)
                        if new_loss < self.model_loss:
                            self.model_loss = new_loss
                            print(self.model)
                            break
                        else:
                            self.model = backup_model

                        continue

                    possible_split = self.test_additive_separability(black_box_node)
                    if len(possible_split) == 1:
                        possible_split = self.test_multiplicative_separability(black_box_node)
                        if len(possible_split) == 1:
                            result = self.create_univariate_node(black_box_node, backup_model)
                            if result:
                                break
                        else:
                            self.split_node(black_box_node, possible_split, split_multiplicatively=True)
                            break  # for now, assume the split works without testing it further beyond the derivative
                    else:
                        self.split_node(black_box_node, possible_split, split_multiplicatively=False)
                        break  # same as above

                    print(self.model)
                else:
                    failed_attempts += 1
                    if failed_attempts == 3:
                        print("couldn't improve model, aborting")
                        break
                    print(f"couldn't improve the model, trying again {3-failed_attempts} times")
                    continue
                failed_attempts = 0

    def test_additive_separability(self, node, cutoff=.5, n_tests=100, check_cliques=False):
        d2ydx = MetaTrainer.get_hessian(node, self.distribution, self.device, n_tests)
        non_zero = abs(d2ydx) > cutoff
        non_zero = non_zero & non_zero.T
        return MetaTrainer.separate_variables_by_component(non_zero)

    def test_multiplicative_separability(self, node, cutoff=.5, n_tests=100, check_cliques=False):
        d2ydx_divf = MetaTrainer.get_hessian(node, self.distribution, self.device, n_tests, divide_by_f=True)
        non_zero = abs(d2ydx_divf) > cutoff
        non_zero = non_zero & non_zero.T
        return MetaTrainer.separate_variables_by_component(non_zero)


if __name__ == '__main__':
    def f(x0, x1, x2):
        return x0 ** 2 + (2.*x1+3.) * (x2+6.)
    distribution = torch.distributions.HalfNormal(torch.ones((3,))*10)
    dataset = generated_dataset.GeneratorDataset(f, distribution, 20000)
    hybrid_child_1 = nn_node.BlackBoxNode(1, [0])
    hybrid_child_2 = nn_node.BlackBoxNode(2, [1, 2])

    hybrid_tree = nn_node.GreyBoxNode(
        operation=operations.MultivariateOperation(operations.MultivariateOp.ADD, False),
        input_set=[0, 1, 2],
        child_nodes=[hybrid_child_1, hybrid_child_2],
        child_input_idxs={hybrid_child_1: [0], hybrid_child_2: [1, 2]},
    )
    utils.load_model(hybrid_tree, "hybrid_tree_100-20220102-114822.pt")
    hybrid_tree.to("cuda")
    # MetaTrainer.get_hessian(hybrid_child_2, distribution, "cuda", divide_by_f=True, n_tests=100)
    # MetaTrainer.get_hessian(hybrid_child_2, distribution, "cuda", divide_by_f=True, n_tests=1000)
    MetaTrainer.get_hessian(hybrid_tree, distribution, "cuda", divide_by_f=False, n_tests=10000)
    # MetaTrainer.get_hessian(hybrid_child_2, distribution, "cuda", divide_by_f=True, n_tests=100000)
    # MetaTrainer.get_hessian(hybrid_child_2, distribution, "cuda", divide_by_f=True, n_tests=1000000)


    # meta_trainer = MetaTrainer(dataset, 3, distribution, model=hybrid_tree)
    # meta_trainer.train(1, skip_initial_training=True)
    pass
