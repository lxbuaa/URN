from torch import optim


def get_optimizer(hyper_para, net):
    optim_name = hyper_para.optimizer.name  # 优化器名字
    optim_lr = hyper_para.optimizer.lr  # 学习率
    optim_b1 = hyper_para.optimizer.b1
    optim_b2 = hyper_para.optimizer.b2
    optim_momentum = hyper_para.optimizer.momentum
    optimizer_dict = {
        "Adam": optim.Adam(net.parameters(), lr=optim_lr, betas=(optim_b1, optim_b2)),
    }
    optimizer = optimizer_dict[optim_name]
    return optimizer

