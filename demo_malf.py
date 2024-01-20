# ******************************************************
# Author           : liuyang && jnulzl
# Last modified    : 2022-12-05 15:26
# Filename         : demo_single.py
# ******************************************************
from __future__ import absolute_import
import argparse
import time

import numpy as np
import torch
import os
import cv2
from core.workspace import create, load_config
from torch.autograd import Variable
from utils.nms.nms_wrapper import nms
from data import anchor_utils
from data.transform.image_util import normalize_img
from data.datasets_utils.visualize import draw_bboxes
from data.anchors_opr.generate_anchors import GeneartePriorBoxes

parser = argparse.ArgumentParser(description='Test Details')
parser.add_argument('--weight_path', default='snapshots/MogFace_Ali-AMS/model_70000.pth',
                    type=str, help='The weight path.')
parser.add_argument('--source', default='', type=str, help='img path')
parser.add_argument('--output_path', default='', type=str, help='save output prediction file')
parser.add_argument('--nms_th', default=0.3, type=float, help='nms threshold.')
parser.add_argument('--pre_nms_top_k', default=5000, type=int, help='number of max score image.')
parser.add_argument('--score_th', default=0.01, type=float, help='score threshold.')
parser.add_argument('--max_bbox_per_img', default=750, type=int, help='max number of det bbox.')
parser.add_argument('--config', '-c', default='./config.yml', type=str, help='config yml.')
parser.add_argument('--test_idx', default=None, type=int)


class DataSetting(object):
    def __init__(self):
        pass


def detect_face(net, image, shrink, generate_anchors_fn):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)

    # print('shrink:{}'.format(shrink))

    width = x.shape[1]
    height = x.shape[0]
    print('width: {}, height: {}'.format(width, height))

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(), volatile=True)

    out = net(x)

    anchors = anchor_utils.transform_anchor((generate_anchors_fn(height, width)))
    anchors = torch.FloatTensor(anchors).cuda()
    decode_bbox = anchor_utils.decode(out[1].squeeze(0), anchors)
    boxes = decode_bbox
    scores = out[0].squeeze(0)

    select_idx_list = []
    tmp_height = height
    tmp_width = width

    test_idx = args.test_idx
    if test_idx is not None:
        for i in range(2):
            tmp_height = (tmp_height + 1) // 2
            tmp_width = (tmp_width + 1) // 2

        for i in range(6):
            if i == 0:
                select_idx_list.append(tmp_height * tmp_width)
            else:
                select_idx_list.append(tmp_height * tmp_width + select_idx_list[i - 1])
            tmp_height = (tmp_height + 1) // 2
            tmp_width = (tmp_width + 1) // 2

        if test_idx == 2:
            boxes = boxes[:select_idx_list[(test_idx - 2)]]
            scores = scores[:select_idx_list[(test_idx - 2)]]
        else:
            boxes = boxes[select_idx_list[test_idx - 3]: select_idx_list[test_idx - 2]]
            scores = scores[select_idx_list[test_idx - 3]: select_idx_list[test_idx - 2]]

    # print('scores shape', scores.shape)
    # print('boxes shape', boxes.shape)
    top_k = args.pre_nms_top_k
    v, idx = scores[:, 0].sort(0)
    idx = idx[-top_k:]
    boxes = boxes[idx]
    scores = scores[idx]

    # [11620, 4]
    boxes = boxes.cpu().numpy()
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    boxes[:, 0] /= shrink
    boxes[:, 1] /= shrink
    boxes[:, 2] = boxes[:, 0] + w / shrink - 1
    boxes[:, 3] = boxes[:, 1] + h / shrink - 1
    # boxes = boxes / shrink
    # [11620, 2]
    scores = scores.cpu().numpy()

    inds = np.where(scores[:, 0] > args.score_th)[0]
    if len(inds) == 0:
        det = np.empty([0, 5], dtype=np.float32)
        return det
    c_bboxes = boxes[inds]
    # [5,]
    c_scores = scores[inds, 0]
    # [5, 5]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

    keep = nms(c_dets, args.nms_th)
    c_dets = c_dets[keep, :]

    max_bbox_per_img = args.max_bbox_per_img
    if max_bbox_per_img > 0:
        image_scores = c_dets[:, -1]
        if len(image_scores) > max_bbox_per_img:
            image_thresh = np.sort(image_scores)[-max_bbox_per_img]
            keep = np.where(c_dets[:, -1] >= image_thresh)[0]
            c_dets = c_dets[keep, :]

    for i in range(c_dets.shape[0]):
        if c_dets[i][0] < 0.0:
            c_dets[i][0] = 0.0

        if c_dets[i][1] < 0.0:
            c_dets[i][1] = 0.0

        if c_dets[i][2] > width - 1:
            c_dets[i][2] = width - 1

        if c_dets[i][3] > height - 1:
            c_dets[i][3] = height - 1

    return c_dets


def process_img(img, net, generate_anchors_fn,  normalize_setting):
    with torch.no_grad():
        img = normalize_img(img.astype(np.float32), normalize_setting)
        max_im_shrink = (0x7fffffff / 200.0 / (
                img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image for caffe
        max_im_shrink = 2.2 if max_im_shrink > 2.2 else max_im_shrink
        shrink = max_im_shrink if max_im_shrink < 1 else 1
        boxes = detect_face(net, img, shrink, generate_anchors_fn)  # origin test
        return boxes


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = load_config(args.config)

    generate_anchors_fn = GeneartePriorBoxes(
        scale_list=cfg['GeneartePriorBoxes']['scale_list'],
        aspect_ratio_list=cfg['GeneartePriorBoxes']['aspect_ratio_list'],
        stride_list=cfg['GeneartePriorBoxes']['stride_list'],
        anchor_size_list=cfg['GeneartePriorBoxes']['anchor_size_list'],
    )

    normalize_setting = DataSetting()
    normalize_setting.use_rgb = cfg['BasePreprocess']['use_rgb']
    normalize_setting.img_mean = (np.array(cfg['BasePreprocess']['img_mean']).astype('float32') * 255)[::-1]
    normalize_setting.img_std = (np.array(cfg['BasePreprocess']['img_std']).astype('float32') * 255)[::-1]
    normalize_setting.normalize_pixel = cfg['BasePreprocess']['normalize_pixel']

    net = create(cfg.architecture)
    # print('Load model from {}'.format(args.weight_path))
    net.load_state_dict(torch.load(args.weight_path))
    net.cuda()
    net.eval()

    onnx_path = args.weight_path.replace('.pth', '.onnx')
    if not os.path.exists(onnx_path):
        img = torch.rand(1, 3, 640, 640)
        model = net.cpu()
        torch.onnx.export(model, img, onnx_path, verbose=True, opset_version=11,
                          input_names=['input'],
                          output_names=['output1', 'output2']
                          )

    print('Finish load model.')
    # For test image list
    if args.source.endswith(".txt"):
        predictions_file_lines = []
        with open(args.source, 'r') as fpR:
            lines = fpR.readlines()
        for line in lines:
            img = cv2.imread(line.strip())
            print('img', line)
            img_show = img.copy()
            t1 = time.time()
            boxes = process_img(img, net, generate_anchors_fn, normalize_setting)
            t2 = time.time()
            print(line)
            print("Inference time : %d" % ((t2 - t1) * 1000))
            predictions_file_lines.append('{}'.format(line.strip()))
            predictions_file_lines.append('{}'.format(len(boxes)))
            for box in boxes:
              predictions_file_lines.append('{} {} {} {} conf {}'.format(box[0], box[1], box[2], box[3], box[4]))
            # img = draw_bboxes(img_show, boxes[:, :4])
            # cv2.imshow("Demo", img)
        
        with open(args.output_path, 'w') as f:
          for line in predictions_file_lines:
            f.write('{}\n'.format(line))
            # if 27 == cv2.waitKey(0):
            #     break
    # For test video
    elif args.source.endswith(".mp4") or args.source.endswith(".avi"):
        cap = cv2.VideoCapture(args.source)
        while True:
            ret, img = cap.read()
            if img is None:
                break
            img_show = img.copy()
            t1 = time.time()
            boxes = process_img(img, net, generate_anchors_fn, normalize_setting)
            t2 = time.time()
            print("Inference time : %d" % ((t2 - t1) * 1000))
            img = draw_bboxes(img_show, boxes[:, :4])
            cv2.imshow("Demo", img)
            if 27 == cv2.waitKey(1):
                break
    # For test single image
    else:
        with torch.no_grad():
            img = cv2.imread(args.source)
            img_show = img.copy()
            t1 = time.time()
            boxes = process_img(img, net, generate_anchors_fn, normalize_setting)
            t2 = time.time()
            print("Inference time : %d" % ((t2 - t1) * 1000))
            print("boxes: ", boxes[0], len(boxes))
            img = draw_bboxes(img_show, boxes[:, :4], output_dir='/content')
            cv2.imwrite("Demo", img)
            cv2.imshow("Demo", img)
            cv2.waitKey(0)