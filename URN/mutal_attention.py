import torch
import torch.nn as nn
import numpy as np
import torchsnooper
from timm.models.layers import DropPath
import torch
import torch.nn.functional as F

from einops import rearrange

from URN.ua import conv

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, activation=True, with_bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.with_bn = with_bn
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class UEMA_Amp(nn.Module):
    def __init__(self, uc_channel, x_channel, is_down_sample=False, alpha=1.0, is_channel=True):
        super().__init__()
        self.is_down_sample = is_down_sample
        self.channel = x_channel
        self.alpha = alpha
        self.uc_preprocess = conv(uc_channel, x_channel, 1, relu=True)
        self.uc_q = nn.Sequential(conv(x_channel, x_channel, 3, relu=True),
                                  conv(x_channel, x_channel, 3, relu=True))
        self.x_k = nn.Sequential(conv(x_channel, x_channel, 1, relu=True),
                                 conv(x_channel, x_channel, 1, relu=True))
        self.x_v = nn.Sequential(conv(x_channel, x_channel, 1, relu=True),
                                 conv(x_channel, x_channel, 1, relu=True))
        self.relu = nn.ReLU(inplace=False)
        self.is_channel = is_channel
        self.channel_wise_conv = nn.Sequential(
            BasicConv2d(x_channel, x_channel, kernel_size=3, padding=1, activation=True, with_bn=True),
            BasicConv2d(x_channel, x_channel, kernel_size=3, padding=1, activation=False, with_bn=False),
        )
        # self.channel_wise_conv = nn.Sequential(
        #     BasicConv2d(x_channel, x_channel, kernel_size=3, padding=1, activation=True, with_bn=True),
        #     BasicConv2d(x_channel, x_channel, kernel_size=3, padding=1, activation=False, with_bn=False),
        # )
        # self.conv_out = conv(x_channel, 1, 1, relu=False)
        # self.conv_out2 = conv(uc_channel + out_channel + noise_channel, out_channel, 3, relu=True)
        # self.conv_out3 = conv(out_channel, out_channel, 3, relu=True)
        # self.conv_out4 = conv(channel, 1, 1)

    # @torchsnooper.snoop()
    def forward(self, x_uc, x):
        if self.is_down_sample:
            x_tmp = nn.AdaptiveAvgPool2d((x.shape[-2] // 2, x.shape[-1] // 2))(x)
        else:
            x_tmp = x.clone()
        b_x, c_x, h_x, w_x = x_tmp.shape
        x_uc_down = F.interpolate(x_uc, size=[h_x, w_x], mode='bilinear', align_corners=True)

        x_uc_alpha = self.uc_preprocess(x_uc_down) * self.alpha

        # get q k v
        uc_q = self.uc_q((x_tmp + x_uc_alpha) / (1.0 + self.alpha)).view(b_x, self.channel, -1).permute(0, 2, 1)
        noise_k = self.x_k(x_tmp).view(b_x, self.channel, -1)
        noise_v = self.x_v(x_tmp).view(b_x, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(uc_q, noise_k)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        x_fuse = torch.bmm(sim, noise_v).permute(0, 2, 1).contiguous().view(b_x, -1, h_x, w_x)
        if self.is_down_sample:
            x_fuse = F.interpolate(x_fuse, size=[h_x * 2, w_x * 2], mode='bilinear', align_corners=True)

        # x_fuse = self.compress_conv(x_fuse)
        if self.is_channel:
            x_fuse = x + self.channel_wise_conv(x_fuse)
        else:
            x_fuse = x + x_fuse
        # x_fuse = self.relu(x_fuse + x)
        return x_fuse
