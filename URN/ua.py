import torch
import torch.nn as nn
import torch.nn.functional as F


class reverse_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(reverse_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 3 if kernel_size == 3 else 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * (torch.sigmoid(map)) + 1

        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out


class simple_attention(nn.Module):
    def __init__(self, in_channel, channel, depth=3, kernel_size=3):
        super(simple_attention, self).__init__()
        self.conv_in = conv(in_channel, channel, 1)
        self.conv_mid = nn.ModuleList()
        for i in range(depth):
            self.conv_mid.append(conv(channel, channel, kernel_size))
        self.conv_out = conv(channel, 1, 1)

    def forward(self, x, map):
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        amap = torch.sigmoid(map)

        x = amap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv_in(x)
        for conv_mid in self.conv_mid:
            x = F.relu(conv_mid(x))
        out = self.conv_out(x)
        out = out + map

        return x, out


class CA(nn.Module):
    def __init__(self, in_channel, channel):
        super(CA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, map):
        b, c, h, w = x.shape

        # compute class probability
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fg = torch.sigmoid(map)

        p = fg - .5

        fg = torch.clip(p, 0, 1)  # foreground
        bg = torch.clip(-p, 0, 1)  # background

        prob = torch.cat([fg, bg], dim=1)

        # reshape feature & prob
        f = x.view(b, h * w, -1)
        prob = prob.view(b, 2, h * w)

        # compute context vector
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)  # b, 2, c

        # k q v compute
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        out = out + map
        return x, out


class UACA(nn.Module):
    def __init__(self, in_channel, channel):
        super(UACA, self).__init__()
        self.channel = channel

        self.conv_query = nn.Sequential(conv(in_channel, channel, 3, relu=True),
                                        conv(channel, channel, 3, relu=True))
        self.conv_key = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                      conv(channel, channel, 1, relu=True))
        self.conv_value = nn.Sequential(conv(in_channel, channel, 1, relu=True),
                                        conv(channel, channel, 1, relu=True))

        self.conv_out1 = conv(channel, channel, 3, relu=True)
        self.conv_out2 = conv(in_channel + channel, channel, 3, relu=True)
        self.conv_out3 = conv(channel, channel, 3, relu=True)
        # self.conv_out4 = conv(channel, 1, 1)

    def forward(self, x, samples):
        b, c, h, w = x.shape

        # 统一尺度
        temp_samples = [F.interpolate(sample, size=x.shape[-2:], mode='bilinear', align_corners=False) for sample in
                        samples]
        # 将samples拼接为向量
        prob = torch.cat(temp_samples, dim=1)

        # 将samples和输入特征的尺寸进行变换，以确保可以相乘
        f = x.view(b, h * w, -1)
        prob = prob.view(b, len(samples), h * w)

        # compute context vector，相乘之后尺寸为(b, len(samples), c)
        context = torch.bmm(prob, f).permute(0, 2, 1).unsqueeze(3)

        # 分别计算q k v
        query = self.conv_query(x).view(b, self.channel, -1).permute(0, 2, 1)
        key = self.conv_key(context).view(b, self.channel, -1)
        value = self.conv_value(context).view(b, self.channel, -1).permute(0, 2, 1)

        # compute similarity map
        sim = torch.bmm(query, key)  # b, hw, c x b, c, 2
        sim = (self.channel ** -.5) * sim
        sim = F.softmax(sim, dim=-1)

        # compute refined feature
        context = torch.bmm(sim, value).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        context = self.conv_out1(context)

        x = torch.cat([x, context], dim=1)
        x = self.conv_out2(x)
        # TODO: 在这里加一下GCN获得的feature
        x = self.conv_out3(x)
        # out = self.conv_out4(x)
        # out = out + map
        return x


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same',
                 bias=False, bn=True, relu=False):
        super(conv, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = conv(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = conv(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
