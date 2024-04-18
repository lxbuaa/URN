import PIL
import cv2
import torch
import torchsnooper
import torchvision
# from PIL.Image import Image
from PIL import Image

from factory.loss_factory import get_loss
import torch.nn.functional as F


def predict(net, batch, hp):
    """
    run on a batch
    Args:
        net: network to be updated
        batch: batch data
        hp: hyperparameters

    Returns: dict of loss values

    """
    batch_loss, seg_loss, cls_loss, edge_loss = 0, 0, 0, 0
    net_pred = net(batch['img'].cuda())
    gt_mask = batch['mask'].unsqueeze(1).cuda().float()

    # pixel-level loss
    if hp.loss.seg.enable:
        seg_loss = get_loss(net_pred.seg.float(), gt_mask, hp.loss.seg)
        batch_loss = batch_loss + seg_loss

    # image-level loss
    if hp.loss.cls.enable:
        cls_loss = get_loss(net_pred.cls.float().view(-1), batch['cls'].cuda().float().view(-1), hp.loss.cls)
        batch_loss = batch_loss + cls_loss

    loss_dict = {
        "batch_loss": batch_loss,
        "seg_loss": seg_loss,
        "cls_loss": cls_loss,
    }

    return loss_dict
