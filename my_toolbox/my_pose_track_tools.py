#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/01/13
    Description:
"""
import time
import argparse

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
import logging
# import Network
from network_MSRA152 import Network
# detector utils
from detector.detector_yolov3 import *  ##

# pose estimation utils
from HPE.dataset import Preprocessing
from HPE.config import cfg
from tfflat.base import Tester
from tfflat.utils import mem_info
from tfflat.logger import colorlogger
# from nms.gpu_nms import gpu_nms
# from nms.cpu_nms import cpu_nms

# import GCN utils
from graph import visualize_pose_matching
from graph.visualize_pose_matching import *

# import my own utils
import sys, os, time

sys.path.append(os.path.abspath("./graph"))
sys.path.append(os.path.abspath("./utils"))
from utils_json import *
from utils_io_file import *
from utils_io_folder import *
from visualizer import *
from visualizer import visualizer
from utils_choose import *
import logging
from sheen import Str, ColoredHandler
from my_toolbox.json_utils import *
from my_toolbox.bipartite_graph import *

from tqdm import tqdm

flag_visualize = True
flag_nms = False  # Default is False, unless you know what you are doing
enlarge_scale = 0.2
nms_method = 'nms'
nms_thresh = 1.
min_scores = 1e-10
min_box_size = 0.
flag_flip = True


def bipartite_graph_matching(current_bbox_dict_list, prev_bbox_dict_list, score_between_two_frames,
                             current_keypoints_dict_list, next_id):
    prev_to_cur_match = Kuhn_Munkras_match(current_bbox_dict_list, prev_bbox_dict_list, score_between_two_frames)
    result_bbox_dict_list = []
    result_keypoints_dict_list = []
    det_number = 0
    assigned_cur_bbox = []
    for prev_index, cur_index in enumerate(prev_to_cur_match):
        if not np.isnan(cur_index):
            assigned_cur_bbox.append(cur_index)
            cur_index = int(cur_index)
            cur_bbox_dict = current_bbox_dict_list[cur_index]
            cur_keypoints_dict = current_keypoints_dict_list[cur_index]
            cur_bbox_dict['det_id'] = det_number
            cur_bbox_dict['track_id'] = prev_index
            cur_keypoints_dict['det_id'] = det_number
            cur_keypoints_dict['track_id'] = prev_index
            result_bbox_dict_list.append(cur_bbox_dict)
            result_keypoints_dict_list.append(cur_keypoints_dict)
            det_number += 1

    # 没有分配track_id的bbox，给其新的track_id
    for cur_index in range(len(current_bbox_dict_list)):
        if cur_index not in assigned_cur_bbox:
            cur_bbox_dict = current_bbox_dict_list[cur_index]
            cur_keypoints_dict = current_keypoints_dict_list[cur_index]
            cur_bbox_dict['det_id'] = det_number
            cur_bbox_dict['track_id'] = next_id
            cur_keypoints_dict['det_id'] = det_number
            cur_keypoints_dict['track_id'] = next_id
            result_bbox_dict_list.append(cur_bbox_dict)
            result_keypoints_dict_list.append(cur_keypoints_dict)
            det_number += 1
            next_id += 1

    return result_bbox_dict_list, result_keypoints_dict_list, next_id


def distance_between_two_boxs(boxA, boxB):
    x1, y1, _, _ = boxA
    x2, y2, _, _ = boxB
    distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
    return distance


def get_track_id_SGCN(bbox_cur_frame, bbox_list_prev_frame, keypoints_cur_frame,
                      keypoints_list_prev_frame):
    assert (len(bbox_list_prev_frame) == len(keypoints_list_prev_frame))

    min_index = None
    min_matching_score = sys.maxsize
    global pose_matching_threshold
    # if track_id is still not assigned, the person is really missing or track is really lost
    track_id = -1

    for det_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        # check the pose matching score
        keypoints_dict = keypoints_list_prev_frame[det_index]
        keypoints_prev_frame = keypoints_dict["keypoints"]
        pose_matching_score = get_pose_matching_score(keypoints_cur_frame, keypoints_prev_frame,
                                                      bbox_cur_frame,
                                                      bbox_prev_frame)

        if pose_matching_score <= pose_matching_threshold and pose_matching_score <= min_matching_score:
            # match the target based on the pose matching score
            min_matching_score = pose_matching_score
            min_index = det_index

    if min_index is None:
        return -1, None
    else:
        track_id = bbox_list_prev_frame[min_index]["track_id"]
        return track_id, min_index


def get_track_id_SpatialConsistency(bbox_cur_frame, bbox_list_prev_frame):
    """ 用当前帧的bbox，去找之前帧中的bboxes的IOU值最大bbox。

        使用一个bbox去前一帧找IOU值最大的。

    """

    thresh = 0.3
    max_iou_score = 0
    max_index = -1

    for bbox_index, bbox_det_dict in enumerate(bbox_list_prev_frame):
        bbox_prev_frame = bbox_det_dict["bbox"]

        boxA = xywh_to_x1y1x2y2(bbox_cur_frame)
        boxB = xywh_to_x1y1x2y2(bbox_prev_frame)
        iou_score = iou(boxA, boxB)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = bbox_index

    if max_iou_score > thresh:
        track_id = bbox_list_prev_frame[max_index]["track_id"]
        return track_id, max_index
    else:
        return -1, None


def get_pose_matching_score(keypoints_A, keypoints_B, bbox_A, bbox_B):
    if keypoints_A == [] or keypoints_B == []:
        print("graph not correctly generated!")
        return sys.maxsize

    if bbox_invalid(bbox_A) or bbox_invalid(bbox_B):
        print("graph not correctly generated!")
        return sys.maxsize

    graph_A, flag_pass_check = keypoints_to_graph(keypoints_A, bbox_A)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    graph_B, flag_pass_check = keypoints_to_graph(keypoints_B, bbox_B)
    if flag_pass_check is False:
        print("graph not correctly generated!")
        return sys.maxsize

    sample_graph_pair = (graph_A, graph_B)
    data_A, data_B = graph_pair_to_data(sample_graph_pair)

    start = time.time()
    flag_match, dist = pose_matching(data_A, data_B)
    end = time.time()
    return dist


def is_target_lost(keypoints, method="max_average"):
    num_keypoints = int(len(keypoints) / 3.0)
    if method == "average":
        # pure average
        score = 0
        for i in range(num_keypoints):
            score += keypoints[3 * i + 2]
        score /= num_keypoints * 1.0
        print("target_score: {}".format(score))
    elif method == "max_average":
        score_list = keypoints[2::3]
        score_list_sorted = sorted(score_list)
        top_N = 4
        assert (top_N < num_keypoints)
        top_scores = [score_list_sorted[-i] for i in range(1, top_N + 1)]
        score = sum(top_scores) / top_N
    if score < 0.6:
        return True
    else:
        return False


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]

    return keep


def iou(boxA, boxB, xyxy=True):
    # box: (x1, y1, x2, y2)
    # determine the (x, y)-coordinates of the intersection rectangle
    if not xyxy:
        # 如果是xy wh那么要转换数据 - xy是最小坐标
        b1_x1, b1_x2 = boxA[0], boxA[0] + boxA[2]
        b1_y1, b1_y2 = boxA[1], boxA[1] + boxA[3]
        b2_x1, b2_x2 = boxB[0], boxB[0] + boxB[2]
        b2_y1, b2_y2 = boxB[1], boxB[1] + boxB[3]
        xA = max(b1_x1, b2_x1)
        yA = max(b1_y1, b2_y1)
        xB = min(b1_x2, b2_x2)
        yB = min(b1_y2, b2_y2)
    else:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    if not xyxy:
        boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
        boxBArea = (boxB[2] + 1) * (boxB[3] + 1)
    else:
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)  # w×h
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)  # w×h

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_bbox_from_keypoints(keypoints_python_data):
    if keypoints_python_data == [] or keypoints_python_data == 45 * [0]:
        return [0, 0, 2, 2]

    num_keypoints = len(keypoints_python_data)
    x_list = []
    y_list = []
    for keypoint_id in range(int(num_keypoints / 3)):
        x = keypoints_python_data[3 * keypoint_id]
        y = keypoints_python_data[3 * keypoint_id + 1]
        vis = keypoints_python_data[3 * keypoint_id + 2]  # 是否可见
        if vis != 0 and vis != 3:
            x_list.append(x)
            y_list.append(y)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)

    if not x_list or not y_list:
        return [0, 0, 2, 2]

    scale = enlarge_scale  # enlarge bbox by 20% with same center position
    bbox = enlarge_bbox([min_x, min_y, max_x, max_y], scale)
    bbox_in_xywh = x1y1x2y2_to_xywh(bbox)
    return bbox_in_xywh


def enlarge_bbox(bbox, scale):
    assert (scale > 0)
    min_x, min_y, max_x, max_y = bbox
    margin_x = int(0.5 * scale * (max_x - min_x))
    margin_y = int(0.5 * scale * (max_y - min_y))
    if margin_x < 0: margin_x = 2
    if margin_y < 0: margin_y = 2

    min_x -= margin_x
    max_x += margin_x
    min_y -= margin_y
    max_y += margin_y

    width = max_x - min_x
    height = max_y - min_y
    if max_y < 0 or max_x < 0 or width <= 0 or height <= 0 or width > 2000 or height > 2000:
        min_x = 0
        max_x = 2
        min_y = 0
        max_y = 2

    bbox_enlarged = [min_x, min_y, max_x, max_y]
    return bbox_enlarged


def inference_keypoints(pose_estimator, test_data):
    cls_dets = test_data["bbox"]
    # nms on the bboxes
    if flag_nms is True:
        cls_dets, keep = apply_nms(cls_dets, nms_method, nms_thresh)
        test_data = np.asarray(test_data)[keep]
        if len(keep) == 0:
            return -1
    else:
        test_data = [test_data]

    # crop and detect pose
    pose_heatmaps, details, cls_skeleton, crops, start_id, end_id = get_pose_from_bbox(pose_estimator,
                                                                                       test_data,
                                                                                       cfg)
    # get keypoint positions from pose
    keypoints = get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id)
    # dump results
    results = prepare_results(test_data[0], keypoints, cls_dets)
    return results


def apply_nms(cls_dets, nms_method, nms_thresh):
    # nms and filter
    keep = np.where((cls_dets[:, 4] >= min_scores) &
                    ((cls_dets[:, 3] - cls_dets[:, 1]) * (
                            cls_dets[:, 2] - cls_dets[:, 0]) >= min_box_size))[0]
    cls_dets = cls_dets[keep]
    if len(cls_dets) > 0:
        if nms_method == 'nms':
            keep = gpu_nms(cls_dets, nms_thresh)
        elif nms_method == 'soft':
            keep = cpu_soft_nms(np.ascontiguousarray(cls_dets, dtype=np.float32), method=2)
        else:
            assert False
    cls_dets = cls_dets[keep]
    return cls_dets, keep


def get_pose_from_bbox(pose_estimator, test_data, cfg):
    cls_skeleton = np.zeros(
        (len(test_data), cfg.nr_skeleton, 3))  # cfg.nr_skeleton=joint number.  size=number*3
    crops = np.zeros((len(test_data), 4))

    batch_size = 1
    start_id = 0
    end_id = min(len(test_data), batch_size)

    test_imgs = []
    details = []
    for i in range(start_id, end_id):
        test_img, detail = Preprocessing(test_data[i], stage='test')
        test_imgs.append(test_img)
        details.append(detail)

    details = np.asarray(details)
    feed = test_imgs
    for i in range(end_id - start_id):
        ori_img = test_imgs[i][0].transpose(1, 2, 0)
        if flag_flip == True:
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
    feed = np.vstack(feed)

    res = pose_estimator.predict_one([feed.transpose(0, 2, 3, 1).astype(np.float32)])[0]
    res = res.transpose(0, 3, 1, 2)

    if flag_flip == True:
        for i in range(end_id - start_id):
            fmp = res[end_id - start_id + i].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            res[i] += fmp
            res[i] /= 2

    pose_heatmaps = res
    return pose_heatmaps, details, cls_skeleton, crops, start_id, end_id


def get_keypoints_from_pose(pose_heatmaps, details, cls_skeleton, crops, start_id, end_id):
    res = pose_heatmaps
    for test_image_id in range(start_id, end_id):
        r0 = res[test_image_id - start_id].copy()
        r0 /= 255.
        r0 += 0.5

        for w in range(cfg.nr_skeleton):
            res[test_image_id - start_id, w] /= np.amax(res[test_image_id - start_id, w])

        border = 10
        dr = np.zeros(
            (cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
        dr[:, border:-border, border:-border] = res[test_image_id - start_id][:cfg.nr_skeleton].copy()

        for w in range(cfg.nr_skeleton):
            dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)

        for w in range(cfg.nr_skeleton):
            lb = dr[w].argmax()
            y, x = np.unravel_index(lb, dr[w].shape)
            dr[w, y, x] = 0
            lb = dr[w].argmax()
            py, px = np.unravel_index(lb, dr[w].shape)
            y -= border
            x -= border
            py -= border + y
            px -= border + x
            ln = (px ** 2 + py ** 2) ** 0.5
            delta = 0.25
            if ln > 1e-3:
                x += delta * px / ln
                y += delta * py / ln
            x = max(0, min(x, cfg.output_shape[1] - 1))
            y = max(0, min(y, cfg.output_shape[0] - 1))
            cls_skeleton[test_image_id, w, :2] = (x * 4 + 2, y * 4 + 2)
            cls_skeleton[test_image_id, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

        # map back to original images
        crops[test_image_id, :] = details[test_image_id - start_id, :]
        for w in range(cfg.nr_skeleton):
            cls_skeleton[test_image_id, w, 0] = cls_skeleton[test_image_id, w, 0] / cfg.data_shape[
                1] * (crops[test_image_id][2] - crops[test_image_id][0]) + crops[test_image_id][0]
            cls_skeleton[test_image_id, w, 1] = cls_skeleton[test_image_id, w, 1] / cfg.data_shape[
                0] * (crops[test_image_id][3] - crops[test_image_id][1]) + crops[test_image_id][1]
    return cls_skeleton


def prepare_results(test_data, cls_skeleton, cls_dets):
    cls_partsco = cls_skeleton[:, :, 2].copy().reshape(-1, cfg.nr_skeleton)

    cls_scores = 1
    dump_results = []
    cls_skeleton = np.concatenate(
        [cls_skeleton.reshape(-1, cfg.nr_skeleton * 3),
         (cls_scores * cls_partsco.mean(axis=1))[:, np.newaxis]],
        axis=1)
    for i in range(len(cls_skeleton)):
        result = dict(image_id=test_data['img_id'],
                      category_id=1,
                      score=float(round(cls_skeleton[i][-1], 4)),
                      keypoints=cls_skeleton[i][:-1].round(3).tolist())
        dump_results.append(result)
    return dump_results


def pose_to_standard_mot(keypoints_list_list, dets_list_list):
    openSVAI_python_data_list = []

    num_keypoints_list = len(keypoints_list_list)
    num_dets_list = len(dets_list_list)
    assert (num_keypoints_list == num_dets_list)

    for i in range(num_dets_list):

        dets_list = dets_list_list[i]
        keypoints_list = keypoints_list_list[i]

        if dets_list == []:
            continue
        img_path = dets_list[0]["imgpath"]
        img_folder_path = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        img_info = {"folder": img_folder_path,
                    "name": img_name,
                    "id": [int(i)]}
        openSVAI_python_data = {"image": [], "candidates": []}
        openSVAI_python_data["image"] = img_info

        num_dets = len(dets_list)
        num_keypoints = len(
            keypoints_list)  # number of persons, not number of keypoints for each person
        candidate_list = []

        for j in range(num_dets):
            keypoints_dict = keypoints_list[j]
            dets_dict = dets_list[j]
            img_id = keypoints_dict["img_id"]
            det_id = keypoints_dict["det_id"]
            track_id = keypoints_dict["track_id"]
            img_path = keypoints_dict["imgpath"]

            bbox_dets_data = dets_list[det_id]
            det = dets_dict["bbox"]
            if det == [0, 0, 2, 2]:
                # do not provide keypoints
                candidate = {"det_bbox": [0, 0, 2, 2],
                             "det_score": 0}
            else:
                bbox_in_xywh = det[0:4]
                keypoints = keypoints_dict["keypoints"]

                track_score = sum(keypoints[2::3]) / len(keypoints) / 3.0

                candidate = {"det_bbox": bbox_in_xywh,
                             "det_score": 1,
                             "track_id": track_id,
                             "track_score": track_score,
                             "pose_keypoints_2d": keypoints}
            candidate_list.append(candidate)
        openSVAI_python_data["candidates"] = candidate_list
        openSVAI_python_data_list.append(openSVAI_python_data)
    return openSVAI_python_data_list


def x1y1x2y2_to_xywh(det):
    x1, y1, x2, y2 = det
    w, h = int(x2) - int(x1), int(y2) - int(y1)
    return [x1, y1, w, h]


def xywh_to_x1y1x2y2(det):
    x1, y1, w, h = det
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]


def bbox_invalid(bbox):
    if bbox == [0, 0, 2, 2]:
        return True
    if bbox[2] <= 0 or bbox[3] <= 0 or bbox[2] > 2000 or bbox[3] > 2000:
        return True
    return False


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # prediction  [image_number,]
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # 前四位
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]  # 1是维度
        # Sort by it
        image_pred = image_pred[(-score).argsort()]  # 将image_pred根据score排序，score越大的预测，排在越前面。
        class_preds = image_pred[:, 5:].max(1, keepdim=True)[1].float()  # keepdim=True  shape : [...,1]
        detections = torch.cat((image_pred[:, :5], class_preds), 1)  # 按列拼，直接拼成它的第5个值。
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # 所有的候选跟置信度最大的比较（也会和它自己比较）
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output
