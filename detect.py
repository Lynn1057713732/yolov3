from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import load_classes, write_results
import argparse
import os
import os.path as osp
from darknet import DarkNet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList(
            [nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


def get_test_input(input_dim, CUDA):
    img = cv2.imread("./imgs/dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
                        "Image / Directory containing images "
                        "to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to"
                        " store detections to", default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions",
                        default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the"
                        " network. Increase to increase accuracy. "
                        "Decrease to increase speed", default="416", type=str)
    parser.add_argument("--scales", dest="scales",
                        help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    # 在不同尺度上进行检测，每个尺度都代表它们在检测不同尺寸的物体时代表它们是否捕获粗糙特征，细粒度特征或其他东西
    scales = args.scales
    # 获取图片地址
    images = args.images
    # batch大小
    batch_size = int(args.bs)
    # 信心分数阀值
    confidence = float(args.confidence)
    # NMS阀值
    nms_thesh = float(args.nms_thresh)
    start = 0
    # 是否有可用的GPU
    CUDA = torch.cuda.is_available()
    # 检测物标种类
    num_classes = 80

    # 导入coco的80个物标种类名字
    classes = load_classes('data/coco.names')

    # 创建神经网络
    print("Loading network.....")
    # 备用配置文件
    model = DarkNet(args.cfgfile)
    # 导入DarkNet网络权重参数
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    # 图片的分辨率，默认为416x416.一般为32的z倍数
    # 输入图像的分辨率，调整这个值可以调节速度与精度之间的折衷
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # 如果有GPU使用，转换模型的数据类型为CUDA
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    """===============检测阶段==============="""
    # 测量时间的检查点
    read_dir = time.time()

    try:
        # 三种图片PNG,JPG,JPEG
        imlist = [osp.join(osp.realpath('.'), images, img) for img in
                  os.listdir(images) if
                  os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[
                      1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = [osp.join(osp.realpath('.'), images)]
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    # 保存检测的目标,det标志指定的检测目录不存在，就创建
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    # load_batch又是一个检查点
    load_batch = time.time()

    # 转换后的图像，我们还维护了一张原始图像列表，以及包含原始图像尺寸的列表im_dim_list
    batches = list(
        map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    # im_batches代表转换后的pytorch的tensor数据类型
    im_batches = [x[0] for x in batches]
    # 原始的数据类型
    orig_ims = [x[1] for x in batches]
    # im_dim_list保存原始图片的维度信息
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        # 转换为CUDA
        im_dim_list = im_dim_list.cuda()

    # 创建batch
    leftover = 0
    if len(im_dim_list) % batch_size:
        leftover = 1
    # 没有整除的情况
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [
            torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                      len(im_batches))])) for i
            in range(num_batches)]

    """按批迭代，生成预测结果，并把执行检测的所有图像的预测结果的张量(它的形状是D x 8,来自write_results函数的输出)连接起来"""
    i = 0

    write = False
    model(get_test_input(inp_dim, CUDA), CUDA)

    start_det_loop = time.time()

    objs = {}

    for batch in im_batches:
        # 加载图片
        start = time.time()  # 开始时间
        if CUDA:
            batch = batch.cuda()

        # 对预测结果应用偏移量
        # 按照YOLO_V3论文中对预测结果进行转换
        # 平滑预测向量
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x
        # (all the boxes)
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

            # prediction = prediction[:,scale_indices]

        """=====将输出结果根据目标分数阈值和非最大值抑制来获得true检测结果====="""
        # 获取信息分数大于阀值的boxes
        # 将相对坐标转换为绝对坐标
        # 对这些boxes执行NMS,然后将结果保存起来
        # 一般将NMS和保存进行分开写，这样逻辑更清晰一些。但是这两个操作都需要循环，所以将这两个操作在一个循环中执行。
        # 毕竟循环比向量化操作效率低得多
        prediction = write_results(prediction, confidence, num_classes,
                                   nms=True, nms_conf=nms_thesh)
        # write_results函数的输出是int（0），意味着没有检测，我们使用continue继续跳过剩下的循环。
        if type(prediction) == int:
            i += 1
            continue

        end = time.time()

        #        print(end - start)

        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(
                imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        if CUDA:
            """torch.cuda.synchronize确保CUDA内核与CPU同步。
            否则，CUDA内核会在GPU作业排队后立即将控制返回给CPU，这时GPU作业尚未完成（异步调用）。
            如果在GPU作业实际结束之前end = time.time（）被打印出来，这可能会导致错误的时间。"""
            torch.cuda.synchronize()

    # 检测是否有检测结果，如果没有就退出程序
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    """
    在绘制边界框之前，我们输出张量中包含的预测是对填充图像的预测，而不是原始图像。
    仅仅将它们重新缩放到输入图像的尺寸并不适用。我们首先需要转换边界框的坐标，
    使得它的测量是相对于填充图像中的原始图像区域。
    """
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(
        -1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(
        -1, 1)) / 2
    """
    现在，我们的坐标的测量是在填充图像中的原始图像区域上的尺寸。
    但是，在函数letterbox_image中，我们通过缩放因子调整了图像的两个维度（记住，这两个维度的调整都用了同一个因子，以保持宽高比）。
    我们现在撤销缩放以获得原始图像上边界框的坐标。
    """
    output[:, 1:5] /= scaling_factor

    # 对那些框边界在图像边界外的边界框进行裁剪。
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,
                                        im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,
                                        im_dim_list[i, 1])

    output_recast = time.time()

    # 如果图像中的边界框太多，将它们全部绘制成同一种颜色可能不大好。
    # 将此文件下载到您的检测器文件夹。这是一个pickle文件，它包含许多可随机选择的颜色。
    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()


    def write(x, batches, results):
        """
        从colors中随机选择的颜色绘制一个矩形框。它还在边界框的左上角创建一个填充的矩形，并将检测到的目标的类写入填充矩形中。
        :param x:
        :param batches:
        :param results:
        :return:
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 3)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        # 使用cv2.rectangle函数的-1参数来创建填充的矩形
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    # 在局部定义write函数，以便它可以访问colors列表。我们也可以将colors作为参数，
    # 但是这会让我们每个图像只能使用一种颜色，这会破坏我们想要使用多种颜色的目的。绘制边界框。
    # 修改了loaded_ims内的图像。
    list(map(lambda x: write(x, im_batches, orig_ims), output))

    # 通过在图像名称前添加“det_”前缀来保存每张图像。我们创建一个地址列表，并把包含检测结果的图像保存到这些地址中。
    det_names = pd.Series(imlist).apply(
        lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

    # 将带有检测结果的图像写入det_names中的地址
    list(map(cv2.imwrite, det_names, orig_ims))

    end = time.time()

    # 打印时间总结。包含哪部分代码需要多长时间才能执行。当我们需要比较不同的超参数如何影响检测器的速度时，这非常有用。
    # 可以在命令行上执行脚本detection.py时设置超参数，如批的大小，目标置信度和NMS阈值（分别通过bs，confidence，nms_thresh标志传递）。
    print()
    print("TIME SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print(
        "{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print(
        "{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)",
                                 output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing",
                                   class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img",
                                   (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    # 清除cuda的缓存
    torch.cuda.empty_cache()
