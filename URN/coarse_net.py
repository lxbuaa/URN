import easydict
import torch.nn as nn

from URN.res_net import My_ResNet50_MCD
from URN.rru_part import RU_up_MCD, outconv


class Coarse_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, dropout_rate=0.5):
        super(Coarse_v2, self).__init__()
        self.res_net = My_ResNet50_MCD(dropout_rate=dropout_rate)
        self.up4 = RU_up_MCD(in_ch=1024, in_ch_skip=512, out_ch=256, dropout_rate=dropout_rate)
        self.up3 = RU_up_MCD(in_ch=256, in_ch_skip=256, out_ch=128, dropout_rate=dropout_rate)
        self.up2 = RU_up_MCD(in_ch=128, in_ch_skip=64, out_ch=64, dropout_rate=dropout_rate)
        self.up1 = RU_up_MCD(in_ch=64, in_ch_skip=0, out_ch=16, with_skip=False, dropout_rate=dropout_rate)
        self.out_conv1 = outconv(16, n_classes)

    # @torchsnooper.snoop()
    def forward(self, x):
        res_net_outs = self.res_net(x)
        x1 = res_net_outs['feature_map'][0]  # (B, 64, 128, 128)
        x2 = res_net_outs['feature_map'][1]  # (B, 256, 64, 64)
        x3 = res_net_outs['feature_map'][2]  # (B, 512, 32, 32)
        x4 = res_net_outs['feature_map'][3]  # (B, 1024, 16, 16)
        out4 = self.up4(x4, x3)
        out3 = self.up3(out4, x2)
        out2 = self.up2(out3, x1)
        out1 = self.up1(out2, None)
        out_x_1 = self.out_conv1(out1)
        return easydict.EasyDict({
            "seg": out_x_1,
            "cls": nn.AdaptiveMaxPool2d(1)(out_x_1)
        })


def get_coarse_net():
    return Coarse_v2()
