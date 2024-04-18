import argparse
import importlib
import os
import time
import torch

from eval import eval_after_train
from factory.bio_data_factory import get_data, get_dataloader
from factory.hp_factory import get_hp
from factory.model_factory import get_model
from factory.optim_factory import get_optimizer
import tqdm
from easydict import EasyDict

from step import predict
from utils.seed_util import set_seed
from configs.config import config_dict as config
from aim import Run


def train_net(net, hp=None, config=None):
    """
    Train our network
    :param net: network to be trained
    :param hp: hyperparameter
    :param config: config file path
    """
    # Choose dataset
    if config is None:
        config = EasyDict()
    if hp is None:
        hp = EasyDict()
    train_loader, train_dataset = get_data(hp, config, 'train')
    test_loaders = get_data(hp, config, 'test')

    # Choose optimizer
    optimizer = get_optimizer(hp, net)

    # Generate log dir
    if not os.path.exists(config.logs_dir):
        os.makedirs(config.logs_dir)

    # Initialize metrics
    torch.cuda.empty_cache()
    current_loss = 1e9
    t_epoch = tqdm.trange(hp.train.epochs)

    # Start to train
    for epoch in t_epoch:
        t_epoch.set_description(f"Epoch, loss={current_loss}")
        t_epoch.refresh()
        time.sleep(0.01)
        epoch_loss, epoch_seg_loss, epoch_cls_loss = 0, 0, 0

        # Begin to step
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            net.train()
            # Train on a batch
            loss_dict = predict(net, batch, hp)
            epoch_loss = epoch_loss + loss_dict['batch_loss'].item()
            loss_dict['batch_loss'].backward()
            optimizer.step()

        current_loss = epoch_loss / (idx + 1)

        # Save checkpoints
        if config.save_checkpoint and (epoch + 1) % config.checkpoint_epochs == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'{config.logs_dir}/checkpoint_{str(epoch)}.pkl')

    eval_after_train(net, test_loaders, hp)

