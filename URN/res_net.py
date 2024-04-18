import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchsnooper
import torchvision
import numpy as np
import ssl
import baal

# from .drop_edge.models import My_GCNModel
# from .mutal_attention.mutal_attention import Mutual_UACA
# from C2F.um_w_net import Coarse_ResNet
# from C2F.unet_model import Unet_MCD
# from .unet_parts import outconv, RRU_down
# from .common_block.bayar_conv import BayarConv
from baal.bayesian import Dropout

ssl._create_default_https_context = ssl._create_unverified_context

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, n_input=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(n_input, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(pretrained=False, layers=[3, 4, 6, 3], backbone='resnet50', n_input=3, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, layers, n_input=n_input, **kwargs)

    pretrain_dict = model_zoo.load_url(model_urls[backbone])
    try:
        model.load_state_dict(pretrain_dict, strict=False)
    except:
        print("loss conv1")
        model_dict = {}
        for k, v in pretrain_dict.items():
            if k in pretrain_dict and 'conv1' not in k:
                model_dict[k] = v
        model.load_state_dict(model_dict, strict=False)
    print("load resnet50 pretrain success")
    return model


class My_ResNet50(nn.Module):
    def __init__(self, pretrained=True, n_input=3):
        """Declare all needed layers."""
        super(My_ResNet50, self).__init__()
        self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def forward(self, x):

        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        feature_map.append(x)  # 128
        x = self.model.maxpool(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)  # 64, 32, 16, 16

        # out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)
        # feature_map.append(out)  # 128
        return feature_map


class My_ResNet50_MCD(nn.Module):
    def __init__(self, pretrained=True, n_input=3, dropout_rate=0.5):
        """Declare all needed layers."""
        super(My_ResNet50_MCD, self).__init__()
        self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
        self.relu = self.model.relu  # Place a hook
        self.dropout = Dropout(p=dropout_rate)

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)，自适应平均池化下采样
        self.fc = nn.Linear(2048, 1)

    # @torchsnooper.snoop()
    def forward(self, x):

        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.dropout(x)
        feature_map.append(x)  # 128
        x = self.model.maxpool(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.dropout(x)
            feature_map.append(x)

        # x_cls = self.avgpool(x)  # 平均池化下采样
        # x_cls = torch.flatten(x_cls, 1)  # 展平处理
        # x_cls = self.fc(x_cls)  # 全连接

        return {
            "feature_map": feature_map,
            # "cls": x_cls
        }

# class My_um_ResNet50(nn.Module):
#     def __init__(self, pretrained=True, n_input=3, n_sample=5, dropout_rate=0.5, with_GNN=True, with_Attn=True):
#         """Declare all needed layers."""
#         super(My_um_ResNet50, self).__init__()
#
#         self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
#         # self.um_model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
#
#         self.relu = self.model.relu  # Place a hook
#
#         # layers_cfg = [4, 5, 6, 7]
#         layers_cfg = [4, 5, 6, 7]
#         self.blocks = []
#         for i, num_this_layer in enumerate(layers_cfg):
#             self.blocks.append(list(self.model.children())[num_this_layer])
#         # self.um_blocks = [list(self.um_model.children())[6], list(self.um_model.children())[7]]
#
#         # for i, num_this_layer in enumerate(layers_cfg):
#         #     self.blocks.append(list(self.model.children())[num_this_layer])
#         self.um_gcn = My_GCNModel(nfeat=5, nclass=1, dropout=0.2, nhidlayer=3, nhid=64 * 64, tensor_size=(64, 64),
#                                   patch_size=(2, 2))
#         # self.fca_down3 = self.um_blocks[0]
#         # self.compress_fca_down3 = outconv(256, 32)
#         # self.fca_down4 = self.um_blocks[1]
#         self.fca_down3 = RRU_down(256 + 1, 256)
#         self.compress_fca_down3 = outconv(256, 32)
#         self.fca_down4 = RRU_down(256 + 512, 256)
#         self.mutual_attn = Mutual_UACA(256, 256, 256)
#         self.bayar_conv = BayarConv(in_channels=256, out_channels=256, padding=2)
#         self.compress_fca_down4 = outconv(256, 64)
#         self.fca_out = outconv(256, 1)
#
#     # @torchsnooper.snoop()
#     def forward(self, x, var, mean):
#         x_tmp = self.model.conv1(x)
#         x_tmp = self.model.bn1(x_tmp)
#         x1 = self.model.relu(x_tmp)  # 16, 64, 128, 128
#         x2 = self.model.maxpool(x1)
#         x2 = self.blocks[0](x2)  # 16, 256, 64, 64
#         x3 = self.blocks[1](x2)  # 16, 512, 32, 32
#         x4 = self.blocks[2](x3)  # 16, 1024, 16, 16
#         # x4 = self.blocks[3](x3) # 16, 2048, 16, 16
#
#         var = F.interpolate(var, size=x2.shape[-2:], mode='bilinear', align_corners=False)
#         x_mean = F.interpolate(mean, size=x2.shape[-2:], mode='bilinear', align_corners=False)
#         x_rgb = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
#         x_gcn = self.um_gcn(x_mean, var, x_rgb)
#
#         x_uc_4 = self.fca_down3(torch.cat((x2, var), dim=1))
#         x_uc_4_compress = self.compress_fca_down3(x_uc_4)
#         x_uc_4_compress = self.relu(x_uc_4_compress + x_gcn * x_uc_4_compress)
#
#         x_uc_5 = self.fca_down4(torch.cat((x_uc_4, x3), dim=1))
#         x_bayar_5 = self.bayar_conv(x_uc_5)
#         x_uc_5 = self.mutual_attn(x_uc_5, x_bayar_5)
#         x_uc_5_compress = self.compress_fca_down4(x_uc_5)
#
#         x_uc_out = self.fca_out(x_uc_5)
#
#         # out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)
#         # feature_map.append(out)  # 128
#         return {
#             "feature_map": [x1, x2, x3, x4],
#             "um_feature_map": [x_uc_4_compress, x_uc_5_compress],
#             "uc_out": x_uc_out,
#             "uc_map": var
#         }
# constrain_features, _ = self.noise_extractor.base_forward(x)
# if __name__ == '__main__':
#     net = My_ResNet50()
#     img = torch.rand((1, 3, 256, 256))
#     fm, out = net(img)
#     print(net(img))
