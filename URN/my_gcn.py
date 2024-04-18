import torch.nn as nn
import torch
import torch.nn.functional as F

from UGNetv2 import ResGCNBlock, DenseGCNBlock, MultiLayerGCNBlock, InecptionGCNBlock, GraphConvolutionBS, \
    Dense


class GraphConvertor(nn.Module):
    def __init__(self, in_channel=3, tensor_size=(256, 256), patch_size=(4, 4)):
        super(GraphConvertor, self).__init__()
        self.num_patch_h = int(tensor_size[0] / patch_size[0])
        self.num_patch_w = int(tensor_size[1] / patch_size[1])

        self.patch_conv_um = PatchConv(kernel_size=patch_size, num_channel=1)
        self.patch_conv_feat = PatchConv(kernel_size=patch_size, num_channel=in_channel)
        self.pos_embed_um = nn.Parameter(torch.zeros(1, 1, self.num_patch_h, self.num_patch_w))
        self.pos_embed_feat = nn.Parameter(torch.zeros(1, in_channel, self.num_patch_h, self.num_patch_w))

        self.node_num = self.num_patch_h * self.num_patch_w
        self.edge_checker = EdgeChecker()

    # @torchsnooper.snoop()
    def forward(self, x_feat, x_var):
        """uncertainty map分块，计算各个patch的确定性"""
        x_var_patch = self.patch_conv_um(
            F.interpolate(x_var, size=x_feat.shape[-2:], mode='bilinear', align_corners=False))
        # x_var_patch = torch.sigmoid(x_var_patch)
        # x_var_patch = torch.sigmoid(x_var_patch)
        x_var_patch = 1.0 - x_var_patch
        x_feat_patch = self.patch_conv_feat(x_feat)
        # 每个batch计算一次node和edge，添加到list中
        node_list = []
        edge_list = []
        B, C, H, W = x_feat_patch.shape
        for b in range(B):
            x_var_b = x_var_patch[b].squeeze(0)
            x_feat_b = x_feat_patch[b].view(-1, C)
            node_list.append(x_feat_b)
            # 通过上、下、左、右移动var矩阵，并做差，即可得到寻找到满足连边规则的边：“连边规则为，每个点向周围确定性比自己低的点连边，边权为确定性之差”
            up_res = self.edge_checker(x_var_b, 'up')
            down_res = self.edge_checker(x_var_b, 'down')
            left_res = self.edge_checker(x_var_b, 'left')
            right_res = self.edge_checker(x_var_b, 'right')
            # up_right_res = self.edge_checker(x_var_b, 'up_right')
            # up_left_res = self.edge_checker(x_var_b, 'up_left')
            # down_right_res = self.edge_checker(x_var_b, 'down_right')
            # down_left_res = self.edge_checker(x_var_b, 'down_left')
            edge_from = torch.cat([
                up_res['from_node'], down_res['from_node'], left_res['from_node'], right_res['from_node'],
                # up_right_res['from_node'], up_left_res['from_node'], down_right_res['from_node'], down_left_res['from_node']
            ], dim=0)
            edge_to = torch.cat([
                up_res['to_node'], down_res['to_node'], left_res['to_node'], right_res['to_node'],
                # up_right_res['to_node'], up_left_res['to_node'], down_right_res['to_node'], down_left_res['to_node']
            ], dim=0)
            edge_dir = torch.stack([edge_from, edge_to], dim=0)
            edge_weight = torch.cat([
                up_res['weight'], down_res['weight'], left_res['weight'], right_res['weight'],
                # up_right_res['weight'], up_left_res['weight'], down_right_res['weight'], down_left_res['weight']
            ], dim=0)
            edge = torch.sparse_coo_tensor(edge_dir, edge_weight, [self.node_num, self.node_num])
            edge_list.append(edge)

        return {
            "edge_list": edge_list,
            "node_list": node_list
        }


class EdgeChecker(nn.Module):
    def __init__(self):
        super(EdgeChecker, self).__init__()
        # self.h = h
        # self.w = w

    def forward(self, x, direction='up'):
        """
        :param x:
        :param direction: 找所有比自己direction方向小的元素的连边
        :return:
        """
        y = torch.zeros(x.shape).cuda()
        if direction == 'up':
            y[:-1, :] = x[1:, :]  # 上移
            dir_shift = (1, 0)
        elif direction == 'down':
            y[1:, :] = x[:-1, :]  # 下移
            dir_shift = (-1, 0)
        elif direction == 'left':
            y[:, :-1] = x[:, 1:]  # 左移
            dir_shift = (0, 1)
        elif direction == 'right':
            y[:, 1:] = x[:, :-1]  # 右移
            dir_shift = (0, -1)
        elif direction == 'up_left':
            y[:-1, :-1] = x[1:, 1:]  # 上移+左移
            dir_shift = (1, 1)
        elif direction == 'up_right':
            y[:-1, 1:] = x[1:, :-1]  # 上移+右移
            dir_shift = (1, -1)
        elif direction == 'down_left':
            y[1:, :-1] = x[:-1, 1:]  # 上移+左移
            dir_shift = (-1, 1)
        elif direction == 'down_right':
            y[1:, 1:] = x[:-1, :-1]  # 上移+右移
            dir_shift = (-1, -1)

        y = y - x
        check_res = torch.where(y > 1e-6)
        to_row = check_res[-2]
        to_col = check_res[-1]
        from_row = check_res[-2] + dir_shift[-2]
        from_col = check_res[-1] + dir_shift[-1]
        # print(to_row, from_row)
        # print(to_col, from_col)
        from_tensor = from_row * x.shape[-2] + from_col
        to_tensor = to_row * x.shape[-1] + to_col
        weight_tensor = y[check_res]
        res_dict = {
            "from_node": from_tensor,
            "from_pos": (from_row, from_col),
            "to_node": to_tensor,
            "to_pos": (to_row, to_col),
            "weight": weight_tensor
        }
        return res_dict


class PatchConv(nn.Module):  # 1*1卷积用来切块，步长=卷积核长
    def __init__(self, kernel_size=(4, 4), num_channel=1):
        super(PatchConv, self).__init__()
        # self.stride = kernel_size[0]
        self.kernel_size = kernel_size
        self.conv_weight_size = (num_channel, num_channel, kernel_size[0], kernel_size[1])
        # self.weights = nn.Parameter(
        #     torch.ones(size=conv_weight_size, dtype=float), requires_grad=False).float().cuda()

    def forward(self, x):
        weights = nn.Parameter(torch.ones(size=self.conv_weight_size, dtype=float), requires_grad=False).float().to(
            x.device)
        return F.conv2d(x, weight=weights, stride=self.kernel_size) / (self.kernel_size[0] * self.kernel_size[1])


class My_GCNModel(nn.Module):
    """
       The model for the single kind of deepgcn blocks.

       The model architecture likes:
       inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                           |------  nhidlayer  ----|
       The total layer is nhidlayer*nbaselayer + 2.
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False,
                 tensor_size=(256, 256),
                 patch_size=(4, 4)):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
        :param inputlayer: the input layer type, can be "gcn", "dense", "none".
        :param outputlayer: the input layer type, can be "gcn", "dense".
        :param nbaselayer: the number of layers in one hidden block.
        :param activation: the activation function, default is ReLu.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param mixmode: enable cpu-gpu mix mode. If true, put the inputlayer to cpu.
        """
        super(My_GCNModel, self).__init__()
        self.graph_convert = GraphConvertor(in_channel=nfeat, tensor_size=tensor_size, patch_size=patch_size)
        self.dropout = dropout
        self.feature_map_size = (int(tensor_size[0] / patch_size[0]), int(tensor_size[1] / patch_size[1]))
        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InecptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, lambda x: x, withbn, withloop)
        else:
            self.outgc = Dense(nhid, nclass, activation)
        self.nclass = nclass
        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()

        self.reset_parameters()

    def reset_parameters(self):
        pass

    # @torchsnooper.snoop()
    def forward(self, x_feat, x_var):
        # 将特征图转为图
        graph = self.graph_convert(x_feat, x_var)
        x_list = []
        for fea, adj in zip(graph['node_list'], graph['edge_list']):
            fea = fea.cuda()
            adj = adj.cuda()
            # print(adj)
            # 初始图卷积层
            x = self.ingc(fea, adj)
            x = F.dropout(x, self.dropout, training=self.training)
            # 中间图卷积层
            for i in range(len(self.midlayer)):
                midgc = self.midlayer[i]
                x = midgc(x, adj)
            # 输出图卷积层
            x = self.outgc(x, adj)
            x = x.view(self.nclass, self.feature_map_size[0], self.feature_map_size[1])  # TODO: 这里reshape的不一定对，重新检查
            # x = x.unsqueeze(0)
            x_list.append(x)
        x_list = torch.stack(x_list, dim=0)  # 图中每个node的分类结果，(B, 1, H, W)
        return x_list


# class MyViG(nn.Module)


if __name__ == '__main__':
    tensor = torch.rand((2, 3, 16, 16)).cuda()
    um = torch.rand((2, 1, 32, 32)).cuda()

    # tensor = torch.tensor([[6, 2, 3],
    #                        [4, 9, 6],
    #                        [7, 8, 9]]).cuda()
    #
    # # 获取张量的尺寸
    # print(tensor.shape)
    e = My_GCNModel(nfeat=3, nclass=3, dropout=0.2, nhidlayer=1, nhid=64, tensor_size=(16, 16),
                    patch_size=(2, 2)).cuda()
    res = e(um, tensor)
    print(res)
