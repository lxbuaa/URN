import torch
import torch.nn as nn
import numpy as np
import torchsnooper
from timm.models.layers import DropPath
import torch
import torch.nn.functional as F

# from ua import conv
# from common_block.bayar_conv import BayarConv
from einops import rearrange

from URN.ua import conv


class MixConversion(nn.Module):
    def __init__(self, in_channel=64, compress_channel=16):
        super().__init__()
        self.conv_compress = torch.nn.Conv2d(kernel_size=1, in_channels=in_channel, out_channels=compress_channel)
        self.bn = torch.nn.BatchNorm2d()
        self.gelu = torch.nn.GELU()
        self.conv_recover = torch.nn.Conv2d(kernel_size=1, in_channels=compress_channel, out_channels=in_channel)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_compress(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.conv_recover(x)
        x = self.sigmoid(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, img_size=256, grid_size_div=16, in_channels=3, hidden_size=768, drop_out=0.1):
        super().__init__()
        grid_size = img_size // grid_size_div
        # grid_size = (grid_size, grid_size)
        img_size = (img_size, img_size)

        # grid_size = img_size / 16
        # patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
        # patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // grid_size) * (img_size[1] // grid_size)

        self.patch_embeddings = torch.nn.Conv2d(in_channels=in_channels,
                                                out_channels=hidden_size,
                                                kernel_size=grid_size,
                                                stride=grid_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))

        self.dropout = torch.nn.Dropout(drop_out)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class MixAttention(nn.Module):
    def __init__(self, attn_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., in_conv=64,
                 out_conv=64, compress_conv=16, depth_in_channel=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads

        self.expend_conv = torch.nn.Conv2d(kernel_size=1, in_channels=depth_in_channel, out_channels=attn_dim)
        self.compress_conv = torch.nn.Conv2d(kernel_size=1, in_channels=attn_dim * 2, out_channels=attn_dim)
        self.scale = qk_scale or head_dim ** -0.5

        self.rgb_q = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(attn_dim, attn_dim)

        self.depth_q = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.depth_k = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.depth_v = nn.Linear(attn_dim, attn_dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(attn_dim, attn_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb_fea, depth_fea):
        b, c, h, w = rgb_fea.shape
        depth_fea = self.expend_conv(depth_fea)
        depth_fea = torch.nn.ReLU()(depth_fea)
        depth_fea = F.interpolate(depth_fea, size=rgb_fea.shape[-2:], mode='bilinear', align_corners=False)
        # noise_fea = self.bayar_conv(rgb_fea).transpose(1, 3).resize(b, w * h, c)
        rgb_fea = rgb_fea.transpose(1, 3).resize(b, w * h, c)
        depth_fea = depth_fea.transpose(1, 3).resize(b, w * h, c)

        B, N, C = rgb_fea.shape

        rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.proj_drop(rgb_fea)

        depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.proj_drop(depth_fea)

        rgb_fea = rgb_fea.transpose(1, 2).reshape(B, C, w, h)
        depth_fea = depth_fea.transpose(1, 2).reshape(B, C, w, h)

        fuse_fea = torch.cat([rgb_fea, depth_fea], dim=1)
        fuse_fea = self.compress_conv(fuse_fea)

        return torch.nn.GELU()(fuse_fea)


class MutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.rgb_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.rgb_proj = nn.Linear(dim, dim)

        self.depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb_fea, depth_fea):
        B, N, C = rgb_fea.shape

        rgb_q = self.rgb_q(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_k = self.rgb_k(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        rgb_v = self.rgb_v(rgb_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        depth_q = self.depth_q(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_k = self.depth_k(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_v = self.depth_v(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # rgb branch
        rgb_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)

        rgb_fea = (rgb_attn @ depth_v).transpose(1, 2).reshape(B, N, C)
        rgb_fea = self.rgb_proj(rgb_fea)
        rgb_fea = self.proj_drop(rgb_fea)

        # depth branch
        depth_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)

        depth_fea = (depth_attn @ rgb_v).transpose(1, 2).reshape(B, N, C)
        depth_fea = self.depth_proj(depth_fea)
        depth_fea = self.proj_drop(depth_fea)

        return rgb_fea, depth_fea


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale * self.dim_scale))
        # x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class Simple_Attention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(F_g)

    # @torchsnooper.snoop()
    def forward(self, g, x):  # g是特征，x是um
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x = F.interpolate(x, size=g.shape[-2:], mode='bilinear', align_corners=False)
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        out = g + g * psi
        # out = self.relu(g + g * psi)
        # out = self.bn(out)
        # 返回加权的 x
        return out


class MixMutualAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.uc_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.uc_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.uc_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.uc_proj = nn.Linear(dim, dim)

        self.noise_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.noise_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.noise_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.depth_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, uc_fea, noise_fea):
        B, N, C = uc_fea.shape
        # uncertainty
        uc_q = self.uc_q(uc_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # uc_k = self.uc_k(uc_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # uc_v = self.uc_v(uc_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q [B, nhead, N, C//nhead]

        # noise
        # noise_q = self.noise_q(noise_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        noise_k = self.noise_k(noise_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        noise_v = self.noise_v(noise_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        uc_attn = (uc_q @ noise_k.transpose(-2, -1)) * self.scale
        uc_attn = uc_attn.softmax(dim=-1)
        uc_attn = self.attn_drop(uc_attn)

        uc_fea = (uc_attn @ noise_v).transpose(1, 2).reshape(B, N, C)
        uc_fea = self.uc_proj(uc_fea)
        uc_fea = self.proj_drop(uc_fea)

        # depth branch
        # noise_attn = (noise_q @ uc_k.transpose(-2, -1)) * self.scale
        # noise_attn = noise_attn.softmax(dim=-1)
        # noise_attn = self.attn_drop(noise_attn)
        #
        # noise_fea = (noise_attn @ uc_v).transpose(1, 2).reshape(B, N, C)
        # noise_fea = self.depth_proj(noise_fea)
        # noise_fea = self.proj_drop(noise_fea)

        return uc_fea


class Mutual_UACA(nn.Module):
    def __init__(self, uc_channel, noise_channel, out_channel):
        super().__init__()
        self.channel = out_channel

        self.uc_q = nn.Sequential(conv(uc_channel, out_channel, 3, relu=True),
                                  conv(out_channel, out_channel, 3, relu=True))
        self.uc_k = nn.Sequential(conv(uc_channel, out_channel, 1, relu=True),
                                  conv(out_channel, out_channel, 1, relu=True))
        self.uc_v = nn.Sequential(conv(uc_channel, out_channel, 1, relu=True),
                                  conv(out_channel, out_channel, 1, relu=True))

        self.noise_q = nn.Sequential(conv(noise_channel, out_channel, 3, relu=True),
                                     conv(out_channel, out_channel, 3, relu=True))
        self.noise_k = nn.Sequential(conv(noise_channel, out_channel, 1, relu=True),
                                     conv(out_channel, out_channel, 1, relu=True))
        self.noise_v = nn.Sequential(conv(noise_channel, out_channel, 1, relu=True),
                                     conv(out_channel, out_channel, 1, relu=True))

        self.conv_out1 = conv(out_channel, out_channel, 3, relu=True)
        self.conv_out2 = conv(uc_channel + out_channel + noise_channel, out_channel, 3, relu=True)
        self.conv_out3 = conv(out_channel, out_channel, 3, relu=True)
        # self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x_uc, x_noise):
        b, c, h, w = x_uc.shape

        # 将samples和输入特征的尺寸进行变换，以确保可以相乘

        # 分别计算q k v
        uc_q = self.uc_q(x_uc).view(b, self.channel, -1).permute(0, 2, 1)
        uc_k = self.uc_k(x_uc).view(b, self.channel, -1)
        uc_v = self.uc_v(x_uc).view(b, self.channel, -1).permute(0, 2, 1)

        noise_q = self.uc_q(x_noise).view(b, self.channel, -1).permute(0, 2, 1)
        noise_k = self.uc_k(x_noise).view(b, self.channel, -1)
        noise_v = self.uc_v(x_noise).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(uc_q, noise_k)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        x_fuse = torch.bmm(sim, noise_v).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        x_fuse = self.conv_out1(x_fuse)

        out = torch.cat([x_uc, x_noise, x_fuse], dim=1)
        out = self.conv_out2(out)
        out = self.conv_out3(out)
        # out = self.conv_out4(x)
        # out = out + map
        return out


class UEMA(nn.Module):
    def __init__(self, uc_channel, x_channel, is_down_sample=False):
        super().__init__()
        self.is_down_sample = is_down_sample
        self.channel = x_channel

        self.uc_q = nn.Sequential(conv(uc_channel, x_channel, 3, relu=True),
                                  conv(x_channel, x_channel, 3, relu=True))
        # self.uc_k = nn.Sequential(conv(uc_channel, out_channel, 1, relu=True),
        #                           conv(out_channel, out_channel, 1, relu=True))
        # self.uc_v = nn.Sequential(conv(uc_channel, out_channel, 1, relu=True),
        #                           conv(out_channel, out_channel, 1, relu=True))

        # self.noise_q = nn.Sequential(conv(noise_channel, out_channel, 3, relu=True),
        #                              conv(out_channel, out_channel, 3, relu=True))
        self.x_k = nn.Sequential(conv(x_channel, x_channel, 1, relu=True),
                                 conv(x_channel, x_channel, 1, relu=True))
        self.x_v = nn.Sequential(conv(x_channel, x_channel, 1, relu=True),
                                 conv(x_channel, x_channel, 1, relu=True))
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(x_channel)
        self.conv_out1 = conv(x_channel, 1, 1, relu=False)
        # self.conv_out2 = conv(uc_channel + out_channel + noise_channel, out_channel, 3, relu=True)
        # self.conv_out3 = conv(out_channel, out_channel, 3, relu=True)
        # self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x_uc, x):

        b_x, c_x, h_x, w_x = x.shape
        if self.is_down_sample:
            x_uc_down = F.interpolate(x_uc, size=[h_x // 2, w_x // 2], mode='bilinear', align_corners=True)
        else:
            x_uc_down = F.interpolate(x_uc, size=[h_x, w_x], mode='bilinear', align_corners=True)
        # 将samples和输入特征的尺寸进行变换，以确保可以相乘

        # 分别计算q k v
        uc_q = self.uc_q(x_uc_down).view(b_x, self.channel, -1).permute(0, 2, 1)
        # uc_k = self.uc_k(x_uc).view(b, self.channel, -1)
        # uc_v = self.uc_v(x_uc).view(b, self.channel, -1).permute(0, 2, 1)

        # noise_q = self.uc_q(x).view(b_x, self.channel, -1).permute(0, 2, 1)
        if self.is_down_sample:
            noise_k = self.x_k(
                F.interpolate(x, size=[x.shape[-1] // 2, x.shape[-2] // 2], mode='bilinear', align_corners=True)).view(
                b_x, self.channel, -1)
            noise_v = self.x_v(
                F.interpolate(x, size=[x.shape[-1] // 2, x.shape[-2] // 2], mode='bilinear', align_corners=True)).view(
                b_x, self.channel, -1).permute(0, 2, 1)
        else:
            noise_k = self.x_k(x).view(b_x, self.channel, -1)
            noise_v = self.x_v(x).view(b_x, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(uc_q, noise_k)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        if self.is_down_sample:
            x_fuse = torch.bmm(sim, noise_v).permute(0, 2, 1).contiguous().view(b_x, -1, h_x // 2, w_x // 2)
            x_fuse = F.interpolate(x_fuse, size=[x_fuse.shape[-1] * 2, x_fuse.shape[-2] * 2], mode='bilinear',
                                   align_corners=True)
        else:
            x_fuse = torch.bmm(sim, noise_v).permute(0, 2, 1).contiguous().view(b_x, -1, h_x, w_x)

        x_fuse = self.relu(x_fuse + x)
        out = self.conv_out1(x_fuse)

        # out = torch.cat([x_uc, x_noise, x_fuse], dim=1)
        # out = self.conv_out2(out)
        # out = self.conv_out3(out)
        # out = self.conv_out4(x)
        # out = out + map
        return {
            "seg": out,
            "feature": x_fuse
        }


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
        # 将samples和输入特征的尺寸进行变换，以确保可以相乘

        x_uc_alpha = self.uc_preprocess(x_uc_down) * self.alpha

        # 分别计算q k v
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
