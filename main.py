import nn_node
import torch
import operations
import generated_dataset

if __name__ == '__main__':
    # black_box_node = nn_node.BlackBoxNode(4)
    # out1 = (black_box_node(*list(map(torch.tensor, [[[1.]], [[2.]], [[3.]], [[4.]]]))))
    # example = torch.tensor([[[-100., 2., 60., 4.]], [[0., 0., 0., 0.]], [[1., 1., 1., 1.]], [[100., 100., 100., 100.]]])
    #
    # operation = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=False)
    # grey_box_node = nn_node.GreyBoxNode(operation, child_nodes=[black_box_node])
    #
    # out2 = (grey_box_node(torch.tensor([[1., 2., 3., 4.]])))
    #
    # operation2 = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=True)
    # grey_box_node_2 = nn_node.GreyBoxNode(operation2, child_nodes=[black_box_node])

    # distribution = torch.distributions.normal.Normal(torch.tensor([0.]), torch.tensor([10.]))
    # dataset = generated_dataset.GeneratorDataset(lambda x: x*2, distribution, 20)
    # for (x, y) in dataset:
    #     print(x, y)

    def f(x0, x1, x2):
        return x0 ** 2 + x1 * x2


    distribution = torch.distributions.normal.Normal(torch.zeros((3,)), torch.ones((3,)))
    dataset = generated_dataset.GeneratorDataset(f, distribution, 2000)

    leaf_nodes = [nn_node.LeafNode(add_linear_layer=False) for _ in range(3)]
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
        child_input_idxs={pow_node: [0], multi_node: [1, 2]}
    )
    print(tree(torch.tensor([[1., 0., 3.]])))
