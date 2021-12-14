import nn_node
import torch
import operations
import generated_dataset

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
    #
    # operation2 = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=True)
    # grey_box_node_2 = nn_node.GreyBoxNode(operation2, child_nodes=[black_box_node])

    distribution = torch.distributions.normal.Normal(torch.tensor([0.]), torch.tensor([10.]))
    dataset = generated_dataset.GeneratorDataset(lambda x: x*2, distribution, 20)
    for (x, y) in dataset:
        print(x, y)
