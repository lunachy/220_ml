import glob
import os
import random
import subprocess

import cv2
import numpy as np
from pathlib import Path
import time

import torch
import torchvision
import torch.nn as nn

from courses import COURSES
import xml.etree.cElementTree as ET


import sys

# from yolov5.detect_api import detect_main
#
# sys.path.append(r"D:\Projects\220_ml\captura\yolov5")



course_names = [i["name"] for i in COURSES[38].values()]


def time_synchronized():
    # pytorch-accurate time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def cvt_color(path):
    for file in Path(path).rglob("*.mp4"):
        s_time = time.time()
        # file = Path(r"D:\Projects\220_ml\captura\订单系统\第2节：百度订单系统度小店课程_tmp.mp4")
        # if file.stem not in course_names: continue
        cap = cv2.VideoCapture(str(file.absolute()))
        isOpened = cap.isOpened  # 判断视频是否可读
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取图像的帧，即该视频每秒有多少张图片
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图像的宽度和高度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('\n', file.name, fps, width, height, frames)
        assert width == 1920 and height == 1080
        continue
        video_tmp_name = f"./我是钱冷门竞价38期/{file.parent.name}/{file.name}"  # 要创建的视频文件名称
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
        video = cv2.VideoWriter(video_tmp_name, fourcc, fps, (width, height))
        while isOpened:
            # 读取每一帧，flag表示是否读取成功，frame为图片的内容
            flag, frame = cap.read()
            if not flag: break
            img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
            video.write(img_bgr)
        video.release()
        # cv2.destroyAllWindows()
        print(time.time() - s_time)
        audio_tmp_name = file.with_suffix(".wav")  # 要创建的视频文件名称
        video_out_name = f"./我是钱冷门竞价38期/{file.parent.name}/{file.stem[:-4] + '.mp4'}"  # 要创建的视频文件名称
        subprocess.call(f"ffmpeg -i {video_tmp_name} -i {audio_tmp_name} {video_out_name}", shell=True)


def video2img():
    for file in Path(r"D:\Projects\220_ml\captura\new1").rglob("*.mp4"):
        s_time = time.time()
        prefix = int(s_time) % 100000
        subprocess.call(f"ffmpeg -i {file} -r 0.3 -q:v 2 -f image2 images_new/{prefix}_%05d.jpg", shell=True)
        print(time.time() - s_time)


def cvr_fmt():
    for file in Path(r"D:\Projects\220_ml\captura\images").glob("*.jpeg"):
        file.rename(file.with_suffix(".jpg"))


def generate_xml(path):
    x, y = 116, 22
    width, height = 1920, 1080
    last_tree = None
    for file in Path(path).glob("*.jpg"):
        xml_file = file.with_suffix(".xml")
        if xml_file.exists():
            tree = ET.ElementTree(file=xml_file)
            tree.find("filename").text = file.name
            tree.find("path").text = str(file)
            xmin = tree.find("object/bndbox/xmin")
            xmax = tree.find("object/bndbox/xmax")
            ymin = tree.find("object/bndbox/ymin")
            ymax = tree.find("object/bndbox/ymax")

            xmax.text = str(min(int(xmin.text) + x, width))
            ymax.text = str(min(int(ymin.text) + y, height))
            tree.write(f'D:/xml/{xml_file.name}')
            last_tree = tree
        else:
            last_tree.find("filename").text = file.name
            last_tree.find("path").text = str(file)
            last_tree.write(f'D:/xml/{xml_file.name}')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
def process_video():
    device = "cpu"
    map_loc = torch.device(device)
    weights = ["best.pt"]
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_loc)['model'].float().fuse().eval())  # load FP32 model
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    model0 = model[-1]
    # model = torch.load("best.pt", map_location=torch.device('cpu'))['model'].float().fuse().eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    out_dir = r"D:\38"
    for file in Path(r"D:\Projects\220_ml\captura\我是钱冷门竞价38期\订单系统").rglob("*tmp.mp4"):
        wav = file.with_suffix(".wav")
        cap = cv2.VideoCapture(str(file))
        isOpened = cap.isOpened  # 判断视频是否可读
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取图像的帧，即该视频每秒有多少张图片
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图像的宽度和高度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('\n', file.name, fps, width, height, frames)
        assert width == 1920 and height == 1080
        video_tmp_name = f"./我是钱冷门竞价38期/{file.parent.name}/{file.name}"  # 要创建的视频文件名称

        while isOpened:
            # 读取每一帧，flag表示是否读取成功，frame为图片的内容
            flag, frame = cap.read()
            if not flag: break
            img = letterbox(frame, new_shape=640)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                # Inference
            t1 = time_synchronized()
            pred = model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.5, 0.5)
            t2 = time_synchronized()

            for i, det in enumerate(pred):  # detections per image
                s = '%gx%g ' % img.shape[2:]  # print string

                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        print(xyxy, conf, cls)
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
                print(f'Done. ({t2 - t1:.3f}s)')
            cv2.imwrite(r"D:\123.jpg", frame)


        # # cv2.destroyAllWindows()
        # audio_tmp_name = file.with_suffix(".wav")  # 要创建的视频文件名称
        # video_out_name = f"./我是钱冷门竞价38期/{file.parent.name}/{file.stem[:-4] + '.mp4'}"  # 要创建的视频文件名称
        # subprocess.call(f"ffmpeg -i {video_tmp_name} -i {audio_tmp_name} {video_out_name}", shell=True)
        #
        # subprocess.call(f"ffmpeg -i {file} -r 0.3 -q:v 2 -f image2 images_new/{prefix}_%05d.jpg", shell=True)


if __name__ == '__main__':
    # detect_main("best.pt", "D:/test")
    # generate_xml(r"D:\Projects\220_ml\captura\images")
    with torch.no_grad():
        process_video()
