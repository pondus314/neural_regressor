from torch.utils.data import DataLoader

import nn_node
import torch
import operations
import generated_dataset
import trainers
import utils

if __name__ == '__main__':
    # black_box_node = nn_node.BlackBoxNode(4)
    # # out1 = (black_box_node(*list(map(torch.tensor, [[[1.]], [[2.]], [[3.]], [[4.]]]))))
    # example = torch.tensor([[[-100., 2., 60., 4.]], [[0., 0., 0., 0.]], [[1., 2., 3., 4.]], [[100., 100., 100., 100.]]])
    # example2 = (torch.tensor([[-100., 2., 60., 4.]]), torch.tensor([[0., 0., 0., 0.]]), torch.tensor([[1., 2., 3., 4.]]), torch.tensor([[100., 100., 100., 100.]]))
    # # print(out1)
    # print(black_box_node(example))
    # print(black_box_node(*example2))
    # operation = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=False)
    # grey_box_node = nn_node.GreyBoxNode(operation, child_nodes=[black_box_node])
    #
    # out2 = (grey_box_node(torch.tensor([[1., 2., 3., 4.]])))

    # operation2 = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=True)
    # grey_box_node_2 = nn_node.GreyBoxNode(operation2, child_nodes=[black_box_node])

    # distribution = torch.distributions.normal.Normal(torch.tensor([0.]), torch.tensor([10.]))
    # dataset = generated_dataset.GeneratorDataset(lambda x: x*2, distribution, 20)
    # for (x, y) in dataset:
    #     print(x, y)

    torch.autograd.set_detect_anomaly(True)

    def f(x0, x1, x2):
        return x0 ** 2 + (2.*x1+3.) * (x2+6.)

    leaf_nodes = [nn_node.LeafNode(add_linear_layer=False) for _ in range(1)] +\
                 [nn_node.LeafNode(add_linear_layer=True) for _ in range(2)]
    pow_operation = operations.UnivariateOperation(operations.UnivariateOp.POWER, False)
    pow_node = nn_node.GreyBoxNode(operation=pow_operation,
                                   child_nodes=leaf_nodes[0:1],
                                   child_input_idxs=None)
    multi_node = nn_node.GreyBoxNode(
        operation=operations.MultivariateOperation(operations.MultivariateOp.MULTIPLY, False),
        child_nodes=leaf_nodes[1:],
        child_input_idxs={leaf_nodes[1]: [0], leaf_nodes[2]: [1]},
    )
    tree = nn_node.GreyBoxNode(
        operation=operations.MultivariateOperation(operations.MultivariateOp.ADD, False),
        child_nodes=[pow_node, multi_node],
        child_input_idxs={pow_node: [0], multi_node: [1, 2]},
        is_root=True,
    )
    distribution = torch.distributions.HalfNormal(torch.ones((3,))*10)
    dataset = generated_dataset.GeneratorDataset(f, distribution, 6000)
    trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

    hybrid_child_1 = nn_node.BlackBoxNode(1)
    hybrid_child_2 = nn_node.BlackBoxNode(2)

    hybrid_tree = nn_node.GreyBoxNode(
        operation=operations.MultivariateOperation(operations.MultivariateOp.ADD, False),
        child_nodes=[hybrid_child_1, hybrid_child_2],
        child_input_idxs={hybrid_child_1: [0], hybrid_child_2: [1, 2]},
        is_root=True
    )

    hybrid_tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]]))

    hybrid_trainer = trainers.ModelTrainer(
        model=hybrid_tree,
        epochs=100,
        lr=0.001,
        max_lr=0.005,
        train_loader=trainloader,
    )

    hybrid_trainer.train()
    hybrid_tree.eval()
    hybrid_tree.cpu()

    black_box = nn_node.BlackBoxNode(3, is_root=True)
    blackbox_trainer = trainers.ModelTrainer(model=black_box, epochs=100, lr=0.001, train_loader=trainloader)

    blackbox_trainer.train()
    black_box.eval()
    black_box.cpu()

    model_trainer = trainers.ModelTrainer(model=tree, epochs=40, lr=0.005, train_loader=trainloader)

    model_trainer.train()
    tree.eval()
    tree.cpu()
    print(list(tree.parameters()))

    print(f(2., 0., 3.), f(1., 2., 7.))
    print(tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]])))
    print(black_box(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]])))
    print(hybrid_tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]])))

    utils.save_model(tree, 'tree')
    utils.save_model(hybrid_tree, 'hybrid_tree')
    utils.save_model(black_box, 'black_box')