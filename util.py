from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from bbox import bbox_iou


def letterbox_image(img, inp_dim):
    """
    使用padding调整图像的大小，保持宽高比一致，并用颜色（128,128,128）填充空白的区域。
    :param img:
    :param inp_dim:
    :return:
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h),
                               interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
    (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    YOLO的输出是一个卷积特征图，它包含沿特征图深度的边界框属性。单元格预测的边界框属性被
    相互堆叠在一起。因此，如果您必须访问（5,6）处单元格的第二个边界框，那么您将不得不通过
    map [5,6，（5 + C）：2 *（5 + C）]对它进行索引。这种形式对输出处理
    （例如根据目标置信度进行阈值处理，向中心坐标添加网格偏移量，应用锚等）非常不方便。
    另一个问题是，由于检测发生在三个尺度上，所以预测图的尺寸将会不同。
    尽管三个特征图的维度不同，但要对它们执行的输出处理操作是相似的。
    最好在单个张量上进行这些操作，而不是三个单独的张量。为了解决这些问题，
    我们引入了函数predict_transform.predict_transform函数将输入的检测特征图转换成二维张量，
    其中张量的每一行对应于边界框属性，按顺序排列
    :param prediction:输出
    :param inp_dim:输入图片尺寸
    :param anchors:锚框大小
    :param num_classes:类别数量
    :param CUDA:可选参数，是否使用gpu
    :return:特征图是B×10647×85形状的张量。B是一个batch中图像的数量，10647是每个图像预测的边界框的数量，85是边界框属性的数量
    """
    batch_size = prediction.size(0)
    # 特征图的步长
    stride = inp_dim // prediction.size(2)
    # 预测特征图的单元格大小
    grid_size = inp_dim // stride
    # 每一个边界框的属性
    # 提示:每个边界框属性为5+C[x,y,w,h,score,p_1,p_1,...,p_c]
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # 锚框的尺寸根据net块的height和width属性。这些属性是输入图像的尺寸，它比检测图大
    # 输入图像是检测图的stride倍.因此，我们必须通过检测特征图的stride来划分锚。
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    # 原始prediction.shape=[1,255,13,13]
    # 转换第一次结果[1,85x3,13x13]--->[1, 255, 169]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors,
                                 grid_size * grid_size)
    # 交换下标1,2.---->[1, 169, 255]
    prediction = prediction.transpose(1, 2).contiguous()
    # 第三次转换--->[1, 13x13x3, 85]=[1,507,85]
    prediction = prediction.view(batch_size, grid_size * grid_size *
                                 num_anchors, bbox_attrs)

    # 使用sigmoid函数处理中心坐标X, Y,和物体的信心分数
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 将网格偏移添加到预测的中心坐标
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.Tensor(a).view(-1, 1)
    y_offset = torch.Tensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # 将锚应用于边界框的尺寸
    anchors = torch.Tensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # 将Sigmoid激活应用于类别分数
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid(
        (prediction[:, :, 5: 5 + num_classes]))

    # 将检测图调整为输入图像的大小
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    """
    将输出结果根据目标分数阈值和非最大值抑制来获得true检测结果
    :param prediction: 预测张量包含B x 10647个边界框的信息
    :param confidence: 置信度
    :param num_classes: 类别数量
    :param nms: 是否有nms操作
    :param nms_conf:NMS IoU阈值
    :return:D x 8的张量。D是所有图像的true检测，每个检测由一行表示。
            每个检测有8个属性，即检测的图像在所属批次中的索引，4个角坐标，目标分数，最大置信度类别的分数以及该类别的索引。
    """
    # 预测张量包含B x 10647个边界框的信息.对于每个具有低于阈值的目标分数的边界框，将它的每个属性（边界框的整个行）的值设置为零
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    # 现在具有的边界框属性由中心坐标以及边界框的高度和宽度描述。但是，使用每个框的一对角点的坐标来计算两个框的IoU更容易。
    # 将框的（中心x，中心y，高度，宽度）属性转换为（左上角x，左上角y，右下角x，右下角y）
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    """
    每幅图像中的true检测结果的数量可能不同。
    例如，批量大小为3，图像1,2和3分别具有5个，2个和4个true检测结果。
    因此，一次只能对一张图像进行置信度阈值和NMS。这意味着，我们不能向量化所涉及的操作
    ，并且必须在prediction的第一维（包含批量中的图像索引）上进行循环。
    """

    batch_size = prediction.size(0)
    image_pred = prediction.new(1, prediction.size(2) + 1)

    # write标志用于指示我们尚未初始化output，我们将使用张量来保存整个批量的true检测结果。
    write = False

    for ind in range(batch_size):
        # 从批次中选择图片
        image_pred = prediction[ind]

        # 每个边界框行有85个属性，其中80个是类别分数。此时，我们只关心具有最大值的类别分数。
        # 从每一行中删除80个类别的分数，并添加具有最大值的类别的索引，以及该类别的类别分数。
        max_conf, max_conf_score = torch.max(
            image_pred[:, 5: (5 + num_classes)], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # 我们已经将具有小于阈值的目标置信度的边界框行设置为零.现在让我们清除它们
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        # 处理一个图像中检测到的类
        try:
            img_classes = unique(image_pred_[:, -1])
        except:
            continue

        # 对每一个检测类进行NMS
        for cls in img_classes:
            # 获取一个特定类的检测结果
            cls_mask = image_pred_ * (
                        image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # 对检测进行结果按大小排序，信息分数是最重要的
            conf_sort_index = torch.sort(image_pred_class[:, 4],
                                         descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # 传进来的参数nms为True
            if nms:
                # １个边界框对其他所有的边界框进行计算IOU,执行NMS
                for i in range(idx):
                    # 获取该次循环中所有我们正则查看boxes的IOUs
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0),
                                        image_pred_class[i + 1:])
                    except ValueError:
                        break

                    except IndexError:
                        break

                    # 每次迭代，任何具有索引大于i的的边界框，
                    # 若其IoU大于阈值nms_thresh（具有由i索引的框），则该边界框将被去除。
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i + 1:] *= iou_mask

                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(
                        image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(
                        -1, 7)
            """
            和以前一样，除非我们有一个检测分配给它，否则我们不会初始化输出张量。
            一旦它被初始化，我们把后续的检测与它连接。我们使用write标志来指示张量
            是否已经初始化。在遍历类的循环结束时，我们将检测结果添加到张量output中。
            """
            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
        # 没有物标种类就输出０
        try:
            return output
        except:
            return 0


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
