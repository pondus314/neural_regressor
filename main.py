import nn_node
import torch

if __name__ == '__main__':
    black_box_node = nn_node.BlackBoxNode(4)
    out = (black_box_node(*list(map(torch.tensor, [[[1.]], [[2.]], [[3.]], [[4.]]]))))
    print(out)
    print(black_box_node(torch.tensor([[[1., 2., 3., 4.]]])))
