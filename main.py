from torch.utils.data import DataLoader

import nn_node
import torch
import operations
import generated_dataset
import trainers
import utils

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    def f(x0, x1, x2):
        return x0 ** 2 + (2.*x1+3.) * (x2+6.)
    distribution = torch.distributions.HalfNormal(torch.ones((3,))*10)
    dataset = generated_dataset.GeneratorDataset(f, distribution, 20000)
    trainloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # leaf_nodes = [nn_node.LeafNode(0, add_linear_layer=False) for _ in range(1)] +\
    #              [nn_node.LeafNode(i + 1, add_linear_layer=True) for i in range(2)]
    # pow_operation = operations.UnivariateOperation(operations.UnivariateOp.POWER, False)
    # pow_node = nn_node.GreyBoxNode(operation=pow_operation,
    #                                input_set=[0],
    #                                child_nodes=leaf_nodes[0:1],
    #                                child_input_idxs=None)
    # multi_node = nn_node.GreyBoxNode(
    #     operation=operations.MultivariateOperation(operations.MultivariateOp.MULTIPLY, False),
    #     input_set=[1,2],
    #     child_nodes=leaf_nodes[1:],
    #     child_input_idxs={leaf_nodes[1]: [0], leaf_nodes[2]: [1]},
    # )
    # tree = nn_node.GreyBoxNode(
    #     operation=operations.MultivariateOperation(operations.MultivariateOp.ADD, False),
    #     input_set=[0, 1, 2],
    #     child_nodes=[pow_node, multi_node],
    #     child_input_idxs={pow_node: [0], multi_node: [1, 2]},
    # )

    # tree_trainer = trainers.ModelTrainer(
    #     model=tree,
    #     epochs=20,
    #     lr=0.005,
    #     max_lr=0.01,
    #     train_loader=trainloader,
    #     show_losses=False,
    #     add_additive_separability_loss=True,
    #     distribution=distribution,
    # )
    #
    # # tree_trainer.train()
    # # utils.save_model(tree, 'tree_20_with_additive_loss')
    # utils.load_model(tree, 'tree_20-20211220-193411.pt')
    # tree.eval()
    # print(list(tree.parameters()))

    hybrid_child_1 = nn_node.BlackBoxNode(1, [0])
    hybrid_child_2 = nn_node.BlackBoxNode(2, [1, 2])

    hybrid_tree = nn_node.GreyBoxNode(
        operation=operations.MultivariateOperation(operations.MultivariateOp.ADD, False),
        input_set=[0, 1, 2],
        child_nodes=[hybrid_child_1, hybrid_child_2],
        child_input_idxs={hybrid_child_1: [0], hybrid_child_2: [1, 2]},
    )

    hybrid_tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]]))

    hybrid_trainer = trainers.ModelTrainer(
        model=hybrid_tree,
        epochs=50,
        lr=0.001,
        max_lr=0.005,
        train_loader=trainloader,
        show_losses=True,
        add_additive_separability_loss=False,
        distribution=distribution,
    )

    # hybrid_trainer.train()
    # utils.save_model(hybrid_tree, 'hybrid_tree_100_no_additive_loss')
    utils.load_model(hybrid_tree, 'hybrid_tree_100-20220102-114822.pt')
    hybrid_tree.eval()

    black_box = nn_node.BlackBoxNode(3)
    blackbox_trainer = trainers.ModelTrainer(
        model=black_box,
        epochs=100,
        lr=0.001,
        max_lr=0.005,
        train_loader=trainloader,
        show_losses=True,
        add_additive_separability_loss=False,
        distribution=distribution,
    )

    # blackbox_trainer.train()
    # utils.save_model(black_box, 'black_box_100_no_additive_loss')
    utils.load_model(black_box, 'black_box_100-20220102-115847.pt')
    black_box.eval()

    print(f(2., 0., 3.), f(1., 2., 7.))
    # print(tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]]).to("cuda")))
    print(hybrid_tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]]).to("cuda")), hybrid_trainer.model_loss)
    print(black_box(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]]).to("cuda")), blackbox_trainer.model_loss)

    meta_trainer = trainers.MetaTrainer(dataset, 3, distribution)
    # meta_trainer.train(1)
    print(meta_trainer.test_additive_separability(black_box))
    print(trainers.MetaTrainer.sample_and_get_hessian(hybrid_child_2, distribution, "cuda", divide_by_f=True))
    print(meta_trainer.test_multiplicative_separability(hybrid_child_2))
