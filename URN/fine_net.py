import easydict
import torch
import torch.nn as nn
from URN.coarse_net import Coarse_v2
from URN.mutal_attention import UEMA_Amp
from URN.my_gcn import My_GCNModel
from URN.res_net import resnet
from URN.rru_part import RU_up, outconv, two_outconv


class Fine_ResNet(nn.Module):
    def __init__(self, coarse_path, n_channels=3, n_classes=1, dropout_rate=0.5, n_sample=5):
        super(Fine_ResNet, self).__init__()
        self.c_net = Coarse_v2(n_channels=3, n_classes=1, dropout_rate=dropout_rate)
        pkl = torch.load(coarse_path, map_location='cpu')['model_state_dict']
        self.c_net.load_state_dict(pkl, strict=True)
        self.n_sample = n_sample
        self.res_net = My_um_ResNet50(dropout_rate=dropout_rate)

        self.up4 = RU_up(in_ch=1024, in_ch_skip=512, out_ch=256)
        self.up3 = RU_up(in_ch=256, in_ch_skip=256, out_ch=128)
        self.up2 = RU_up(in_ch=128, in_ch_skip=64, out_ch=64)
        self.up1 = RU_up(in_ch=64, in_ch_skip=0, out_ch=16, with_skip=False)

        self.uema_4 = UEMA_Amp(1, 256, is_down_sample=False, is_channel=True)
        self.uema_3 = UEMA_Amp(1, 128, is_down_sample=False, is_channel=True)
        self.uema_2 = UEMA_Amp(1, 64, is_down_sample=True, is_channel=True)

        self.out_conv1 = outconv(16, n_classes)
        self.out_conv2 = two_outconv(64)
        self.out_conv3 = two_outconv(128)
        self.out_conv4 = two_outconv(256)

        self.compress_conv = outconv(4, 3)

    # @torchsnooper.snoop()
    def forward(self, x):
        self.c_net.cuda()
        with torch.no_grad():
            sample_list = [self.c_net(x)['seg'] for i in range(self.n_sample)]
            var_sample = torch.var(torch.stack(sample_list, dim=0), keepdim=True, axis=0)
            var = var_sample.squeeze(0).cuda()
            mean_sample = torch.mean(torch.stack(sample_list, dim=0), keepdim=True, axis=0)
            mean = mean_sample.squeeze(0).cuda()
        self.c_net.cpu()
        var = torch.sigmoid(var)
        x = self.compress_conv(torch.cat([x, mean], dim=1))
        res_net_outs = self.res_net(x, var)
        x1 = res_net_outs['feature_map'][0]  # (B, 64, 128, 128)
        x2 = res_net_outs['feature_map'][1]  # (B, 256, 64, 64)
        x3 = res_net_outs['feature_map'][2]  # (B, 512, 32, 32)
        x4 = res_net_outs['feature_map'][3]  # (B, 1024, 16, 16)
        out4 = self.up4(x4, x3)  # B, 256, 32, 32
        uema4 = self.uema_4(var, out4)
        out_4 = self.out_conv4(uema4)
        out3 = self.up3(uema4, x2)  # B, 128, 64, 64
        uema3 = self.uema_3(var, out3)
        out_3 = self.out_conv3(uema3)
        out2 = self.up2(uema3, x1)  # B, 64, 128, 128
        uema2 = self.uema_2(var, out2)
        out_2 = self.out_conv2(uema2)
        out1 = self.up1(uema2, None)  # B, 16, 256, 256
        out_x_1 = self.out_conv1(out1)  # B, 1, 256, 256

        return easydict.EasyDict({
            "seg": out_x_1,
            "cls": nn.AdaptiveMaxPool2d(1)(out_x_1),
            "seg_deep": [out_2, out_3, out_4],
            "uc": var
        })


class My_um_ResNet50(nn.Module):
    def __init__(self, pretrained=True, n_input=3, dropout_rate=0.5):
        """Declare all needed layers."""
        super(My_um_ResNet50, self).__init__()
        self.model = resnet(n_input=n_input, pretrained=pretrained, layers=[3, 4, 6, 3], backbone='resnet50')
        self.relu = self.model.relu  # Place a hook
        layers_cfg = [4, 5, 6]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])
        self.bn_first = nn.BatchNorm2d(64)
        self.um_gcn_first = My_GCNModel(nfeat=3, nclass=64, dropout=0.20, nhidlayer=1, nhid=64, tensor_size=(256, 256),
                                        patch_size=(2, 2))

        self.bn = nn.ModuleList([nn.BatchNorm2d(256), nn.BatchNorm2d(512), nn.BatchNorm2d(1024)])
        self.um_gcn = nn.ModuleList([
            My_GCNModel(nfeat=64, nclass=256, dropout=0.15, nhidlayer=1, nhid=256, tensor_size=(64, 64),
                        patch_size=(1, 1)),
            My_GCNModel(nfeat=256, nclass=512, dropout=0.10, nhidlayer=1, nhid=512, tensor_size=(64, 64),
                        patch_size=(2, 2)),
            My_GCNModel(nfeat=512, nclass=1024, dropout=0.05, nhidlayer=1, nhid=1024, tensor_size=(32, 32),
                        patch_size=(2, 2))
        ])

        self.relu = nn.ReLU(inplace=False)

    # @torchsnooper.snoop()
    def forward(self, x, x_uc):
        feature_map = []
        """B 64 H/2 H/2"""
        x = self.relu(self.model.relu(self.model.bn1(self.model.conv1(x))) + self.um_gcn_first(x, x_uc))
        feature_map.append(x)  # 128
        """B 64 H/4 H/4"""
        x = self.model.maxpool(x)
        for i, block in enumerate(self.blocks):
            """B 256 H/4 W/4"""
            """B 512 H/8 W/8"""
            """B 1024 H/16 W/16"""
            x = self.relu(block(x) + self.um_gcn[i](x, x_uc))
            feature_map.append(x)

        return {
            "feature_map": feature_map,
        }


def get_fine_net(dataset_name):
    network = Fine_ResNet(dataset_name=dataset_name)
    return network

