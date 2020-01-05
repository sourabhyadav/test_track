#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2019/12/18
"""
import numpy as np

# copy from DetectAndTrack  https://github.com/facebookresearch/DetectAndTrack/blob/master/lib/convert/box.py

expand_ratio = 1.2
keypoint_num = 15


def expand_boxes(boxes, ratio):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= ratio
    h_half *= ratio

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def compute_boxes_from_pose(poses):
    """
    Args:
        poses (list of list of list of floats):
            list of poses in each frame, each list contains list of poses in
            that frame, where each pose is a 15*3 element list (MPII style).
    Returns:
        boxes: (list of list of list of floats):
            list of boxes in each frame, each list contains a list of boxes in
            that frame, where each pose is [x, y, w, h] list.
    Added by rgirdhar
    """
    boxes = []

    frame_poses_np = np.array(poses)
    # only consider the points that are marked "1", i.e. labeled and visible
    valid_pts = frame_poses_np[:, 2] == 1
    valid_pose = frame_poses_np[valid_pts, :]
    if valid_pose.shape[0] == 0:
        return [0, 0, 2, 2]
    box = np.array([
        np.min(valid_pose[:, 0]),
        np.min(valid_pose[:, 1]),
        # The +1 ensures the box is at least 1x1 in size. Such
        # small boxes will be later removed anyway I think
        np.max(valid_pose[:, 0]) + 1,
        np.max(valid_pose[:, 1]) + 1,
    ])
    box = expand_boxes(np.expand_dims(box, 0), expand_ratio)[0]
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]
