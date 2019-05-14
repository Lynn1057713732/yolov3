from __future__ import division

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from util import predict_transform


def parse_cfg(cfgfile):
    """
    解析configurstion file
    :param cfgfile:
    :return: blocks列表。列表中每个block是一个字典，里面是网络的构建参数
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.rstrip().lstrip() for x in lines]

    blocks = []
    block = {}

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()

        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)
    print('\n\n'.join([repr(x) for x in blocks]))
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    """自定义上采样类，可自由灵活使用"""
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        b = x.data.size(0)
        c = x.data.size(1)
        h = x.data.size(2)
        w = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(b, c, h, 1, w, 1).expand(
            b, c, h, stride, w, stride).contiguous().view(b, c, h * stride,
                                                          w * stride)
        return x


class MaxPoolStride1(nn.Module):
    """自定义池化类"""
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(input=x, pad=[0, self.pad, 0, self.pad],
                         mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    # def forward(self, x, inp_dim, num_classes, confidence):
    #     x = x.data
    #     global CUDA
    #     prediction = x
    #     prediction = predict_transform(prediction, inp_dim, self.anchors,
    #                                    num_classes, confidence, CUDA)
    #     return prediction


def create_modules(blocks):
    """
    根据blocks创建模块
    :param blocks: 列表里面是字典
    :return: net_info是网络结构信息，module_list创建出来的模块
    """
    # 每一块的卷积核数量，初始化
    filters = None
    # cfg中的网络信息
    net_info = blocks[0]

    # 包含nn.Module对象的普通列表
    module_list = nn.ModuleList()

    # 索引块有助于实现路线
    index = 0
    # 用来跟踪前一层卷积核的深度(过滤器的数量)，初始RGB3层。
    # 新生成的卷积层的高度和宽度可以再cfg中拿到
    prev_filters = 3
    # 将每个块的输出过滤器数添加到列表中，之后的route的特征图是使用来自前面的层
    output_filters = []

    # ==========迭代块列表，为每一个快创建一个pytorch模块==========
    for x in blocks:
        module = nn.Sequential()
        # 网络参数层，实际并不参与神经网络传播
        if x["type"] == "net":
            continue

        # 卷积层
        elif x["type"] == "convolutional":
            """从层中获取信息"""
            # 激活函数
            activation = x["activation"]
            try:
                # 是否归一化
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except Exception as e:
                batch_normalize = 0
                bias =True

            # 卷积核数量
            filters = int(x["filters"])
            # 卷积核尺寸
            kernel_size = int(x["size"])
            # 步长
            stride = int(x["stride"])
            # padding
            padding = int(x["pad"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # 增加卷积层
            conv = nn.Conv2d(in_channels=prev_filters, out_channels=filters,
                             kernel_size=kernel_size, stride=stride,
                             padding=pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # 增加归一化层
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # 确认激活函数
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        # 上采样层：使用Bilinear2dUpsampling双线性上采样，层上采样２倍
        elif x["type"] == "upsample":
            stride = int(x["stride"])
            # 可使用自定义的采样类
            # upsample = Upsample(stride)
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        # 如果是一个route层
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")

            # route开始
            start = int(x["layers"][0])
            # route结束<如果只有一个参数>
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # 激活标注
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[
                    index + end]
            else:
                filters = output_filters[index + start]

        # shortcut(跳过连接)
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(kernel_size=size, stride=stride)
            else:
                # 自定义池化类
                maxpool = MaxPoolStride1(size)

            module.add_module("maxpool_{}".format(index), maxpool)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            # 锚框类型选择
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            # 定义新的层，用于检测边界框的锚
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        else:
            print("Something cfg wrong")
            assert False

        """一次循环结束时的操作"""
        # 添加单个块的module到列表里
        module_list.append(module)
        # 记录这一块module的卷积核数量
        prev_filters = filters
        # 再整个卷积核数量列表里添加这一块卷积核数量
        output_filters.append(filters)

        index += 1

    return net_info, module_list


class DarkNet(nn.Module, ABC):
    """真正的检测器网络"""
    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile=cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        """
        两个作用：1.计算输出　2.为了处理更方便，对输出的检测特征图进行变换
        :param x:输入
        :param CUDA:是否使用GPU加速训练，如果为True,就是用GPU
        :return:
        """
        detections = []  # 最后的预测结果列表
        modules = self.blocks[1:]  # 第一个层是net,和计算无关
        # route和shortcut层需要前面的层的输出图，因此我们将每个层的输出特征图缓存在字典outputs中
        outputs = {}  # 键是层的索引，值是特征图

        write = 0
        for i in range(len(modules)):
            module_type = (modules[i]["type"])
            if module_type == "convolutional" or module_type == "upsample" or \
                    module_type == "maxpool":
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == "route":
                layers = modules[i]["layers"]
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                        map1 = outputs[i + layers[0]]
                        map2 = outputs[i + layers[1]]

                        x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(modules[i]["from"])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                # 获取输入维度信息
                inp_dim = int(self.net_info["height"])

                # 获取类别的数量
                num_classes = int(modules[i]["classes"])

                # 输出结果
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                if type(x) == int:
                    continue
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

                outputs[i] = outputs[i - 1]


                try:
                    return detections
                except:
                    return 0

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        # 权重文件的前160个字节存储5个int32值，它们构成文件的头部
        # 1.主版本号
        # 2.小版本号
        # 3.上版本号
        # 4,5图片在训练期间看到的网络
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # 剩下的字节按上文中提到的顺序表示权重。
        # 权重存储为float32或32位浮点数。把权重加载到一个np.ndarray中
        weights = np.fromfile(fp, dtype=np.float32)

        # 遍历权重文件，并将权重加载到我们网络的模块中。
        # 名为ptr的变量来跟踪我们在权重数组中的位置
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            # 首先检查convolutional块是有存在值为True的batch_normalize
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # 获取归一化层的权重
                    num_bn_biases = bn.bias.numel()

                    # 加载权重
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # 将加载到的权重注入模型的权重中
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # 复制数据到模型中
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # batch_norm不为True，只需加载卷积层的偏置
                    # 偏置项的数量
                    num_biases = conv.bias.numel()

                    # 加载权重
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # 通过模型权重参数的维度来重塑加载权重参数
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # 复制数据
                    conv.bias.data.copy_(conv_biases)

                # 最后加载卷积层权重
                num_weights = conv.weight.numel()

                # 和上面相同的处理权重
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
