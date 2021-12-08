import nn_node
import torch
import operations

if __name__ == '__main__':
    black_box_node = nn_node.BlackBoxNode(4)
    out1 = (black_box_node(*list(map(torch.tensor, [[[1.]], [[2.]], [[3.]], [[4.]]]))))
    print(out1)
    print(black_box_node(torch.tensor([[[1., 2., 3., 4.]], [[0., 0., 0., 0.]], [[1., 1., 1., 1.]], [[100., 100., 100., 100.]]])))

    operation = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=False)
    grey_box_node = nn_node.GreyBoxNode(operation, child_nodes=[black_box_node])

    out2 = (grey_box_node(torch.tensor([[1., 2., 3., 4.]])))
    print(out2)

    operation2 = operations.UnivariateOperation(operations.UnivariateOp.POWER, add_linear_layer=True)
    grey_box_node_2 = nn_node.GreyBoxNode(operation2, child_nodes=[black_box_node])
    print(grey_box_node_2(torch.tensor([[[1., 2., 3., 4.]], [[0., 0., 0., 0.]], [[1., 1., 1., 1.]], [[100., 100., 100., 100.]]])))
