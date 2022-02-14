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

    try:
        utils.load_model(hybrid_tree, 'hybrid_tree.pt')
    except:
        hybrid_trainer.train()
        utils.save_model(hybrid_tree, 'hybrid_tree.pt')
    hybrid_tree.eval()

    black_box = nn_node.BlackBoxNode(3)
    blackbox_trainer = trainers.ModelTrainer(
        model=black_box,
        epochs=50,
        lr=0.001,
        max_lr=0.005,
        train_loader=trainloader,
        show_losses=True,
        add_additive_separability_loss=False,
        distribution=distribution,
    )

    try:
        utils.load_model(black_box, 'black_box.pt')
    except:
        blackbox_trainer.train()
        utils.save_model(black_box, 'black_box.pt')
    black_box.eval()

    # print("Equation f(2,0,3); f(1,2,7)")
    # print(f(2., 0., 3.), f(1., 2., 7.))
    # print("Hybrid f(2,0,3); f(1,2,7)")
    # print(hybrid_tree(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]])))
    # print("Black Box f(2,0,3); f(1,2,7)")
    # print(black_box(torch.tensor([[[2., 0., 3.]], [[1., 2., 7.]]])))

    meta_trainer = trainers.MetaTrainer(dataset, 3, distribution)
    # meta_trainer.train(1)
    # print(meta_trainer.test_additive_separability(black_box))
    print(trainers.MetaTrainer.get_hessian(hybrid_child_2, distribution, "cpu", divide_by_f=True))
    print(meta_trainer.test_multiplicative_separability(hybrid_child_2))

    def func(x):
        return x[:,0] ** 2 + (2.*x[:,1]+3.) * (x[:,2]+6.)

    def func_branch(x):
        return (2.*x[:,0]+3.) * (x[:,1]+6.)

    def deriv(y, x):
        return torch.autograd.grad(y.sum(), x, create_graph=True)[0]

    def diff(dydx, x):
        return torch.stack([torch.autograd.grad(dydx[:, i].sum(), x, create_graph=True)[0] for i in range(n)], dim=2)

    x = torch.tensor([[2.,0.,3.], [1.,2.,7.]], requires_grad=True)

    y_eqn = func(x)
    y_hyb = hybrid_tree(x)

    y_eqn_branch = func_branch(x[:,1:])
    y_hyb_branch = hybrid_child_2(x[:,1:])[:,0,0]


    dy_eqn = deriv(y_eqn, x)
    dy_hyb = deriv(y_hyb, x)

    dy_eqn_branch = deriv(y_eqn_branch, x)
    dy_hyb_branch = deriv(y_hyb_branch, x)

    print("y_eqn", y_eqn)
    print("y_hyb", y_hyb)
    print("y_eqn_branch", y_eqn_branch)
    print("y_hyb_branch", y_hyb_branch)
    print("dy_eqn", dy_eqn)
    print("dy_hyb", dy_hyb)
    print("dy_eqn_branch", dy_eqn_branch)
    print("dy_hyb_branch", dy_hyb_branch)

    from Visualiser2D import *
    from functools import partial

    def diff_func_branch(x):
        x.requires_grad = True
        y = func_branch(x)
        return deriv(y,x)[:,1]

    def diff_hyb_branch(x):
        x.requires_grad = True
        y = hybrid_child_2(x)
        return deriv(y,x)[:,1]

    def diff_func_branch_div(x):
        x.requires_grad = True
        y = func_branch(x)
        return deriv(y,x)[:,1] / y

    def diff_hyb_branch_div(x):
        x.requires_grad = True
        y = hybrid_child_2(x)[:,0,0] + 186
        return deriv(y,x)[:,1] / y

    vis = Visualiser2D()

    # vis.update(func, step=0.05, lim=[0,10,0,10], transparent=True, dim=3, other_dim_val=5)
    # vis.update(hybrid_tree, step=0.05, lim=[0,10,0,10], transparent=False, dim=3, other_dim_val=5)

    # vis.update(func_branch, step=0.05, lim=[0,10,0,10], transparent=True)
    # vis.update(hybrid_child_2, step=0.05, lim=[0,10,0,10], transparent=False)

    # vis.update(diff_func_branch, step=0.05, lim=[0,10,0,10], transparent=True)
    # vis.update(diff_hyb_branch, step=0.05, lim=[0,10,0,10], transparent=False)

    vis.update(diff_func_branch_div, step=0.05, lim=[0,10,0,10], transparent=True)
    vis.update(diff_hyb_branch_div, step=0.05, lim=[0,10,0,10], transparent=False)

    import pdb; pdb.set_trace()
