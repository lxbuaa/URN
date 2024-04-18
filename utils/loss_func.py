import numpy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class"""

    def __init__(self, smooth=1, p=2, pos_weight=None):
        """
        Args:
            smooth: A float number to smooth loss, and avoid NaN error, default: 1
            p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        """
        Args:
            predict: A tensor of shape [N, *]
            target: A tensor of shape same with predict
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction

        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.sigmoid(predict).contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target)) * 2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1.0 - dice
        return loss


class MyBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0):
        """
        将所有BCE都换成这个，统一起来，预先sigmoid
        :param pos_weight:
        """
        super(MyBCELoss, self).__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).cuda())

    def forward(self, predict, target):
        return self.bce(predict, target)