#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/02/03
    Description: 利用未来帧gt的信息，从未来回到过去进行矫正。
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################
##单纯为了Debug
image_crop_output_path = '/media/D/light-track/data/demo/crop'
image_seed_crop_output_path = '/media/D/light-track/data/demo/seed_crop'
tracking_gt_info = []

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(ColoredHandler())


################

def initialize_parameters():
    # global video_name, img_id

    global nms_method, nms_thresh, min_scores, min_box_size
    nms_method = 'nms'
    nms_thresh = 1.
    min_scores = 1e-10
    min_box_size = 0.

    global keyframe_interval, enlarge_scale, pose_matching_threshold
    keyframe_interval = 40  # choice examples: [2, 3, 5, 8, 10, 20, 40, 100, ....]

    enlarge_scale = 0.2  # how much to enlarge the bbox before pose estimation
    pose_matching_threshold = 0.5

    global flag_flip
    flag_flip = True

    global total_time_POSE_ESTIMATOR, total_time_POSE_SIMILARITY, total_time_DET, total_time_ALL, total_time_ASSOCIATE
    global total_num_FRAMES, total_num_PERSONS, total_num_VIDEOS
    total_time_POSE_ESTIMATOR = 0
    total_time_POSE_SIMILARITY = 0
    total_time_DET = 0
    total_time_ALL = 0
    total_time_ASSOCIATE = 0
    total_num_VIDEOS = 0
    total_num_FRAMES = 0
    total_num_PERSONS = 0

    """test"""
    global filter_bbox_number, iou_alpha1, pose_alpha1
    filter_bbox_number = 0
    iou_alpha1 = 1.5
    pose_alpha1 = -0.95  # 求的是pose差异值，差异值越小表示越越相似。

    return


def light_track(pose_estimator,
                image_folder, output_json_path,
                visualize_folder, output_video_path, gt_info):
    global total_time_POSE_ESTIMATOR, total_time_POSE_SIMILARITY, total_time_DET, total_time_ALL, total_time_ASSOCIATE
    global video_name, iou_alpha1, pose_alpha1
    global filter_bbox_number, total_num_FRAMES, total_num_PERSONS, total_num_VIDEOS
    ''' 1. statistics: get total time for lighttrack processing'''
    st_time_total = time.time()
    ### hyper-papermet
    keypoints_number = 15
    interval = 5

    bbox_dets_list_list = []
    keypoints_list_list = []

    num_imgs = len(gt_info)

    first_img_id = 0

    start_from_labeled = False
    if start_from_labeled:
        first_img_id = find_first_labeled_opensvai_json(gt_info)

    next_id = 0  # track_id 从0开始算
    img_id = first_img_id
    total_num_FRAMES += num_imgs

    gt_frame_index_list = find_gt_frame_index_list(gt_info, interval=interval)

    logger.info("video_name:{}".format(video_name))
    while img_id < num_imgs:
        ## loop Initialization
        img_gt_info = gt_info[img_id]
        image_name, labeled, candidates_info = read_image_data_opensvai_json(img_gt_info)
        img_path = os.path.join(image_folder, image_name)

        bbox_dets_list = []  # keyframe: start from empty
        keypoints_list = []  # keyframe: start from empty
        prev_frame_img_id = max(0, img_id - first_img_id - 1)

        # 假如第一帧是gt帧，那么直接复制gt的结果，放到list_list中
        if start_from_labeled and img_id == first_img_id:
            num_dets = len(candidates_info)
            for det_id in range(num_dets):
                track_id, bbox_det, keypoints = get_candidate_info_opensvai_json(candidates_info, det_id)
                # first帧直接使用
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": det_id,
                                 "imgpath": img_path,
                                 "track_id": track_id,
                                 "bbox": bbox_det}
                keypoints_dict = {"img_id": img_id,
                                  "det_id": det_id,
                                  "imgpath": img_path,
                                  "track_id": track_id,
                                  "keypoints": keypoints}
                bbox_dets_list.append(bbox_det_dict)
                keypoints_list.append(keypoints_dict)
                next_id = max(next_id, track_id)
                next_id += 1
            bbox_dets_list_list.append(bbox_dets_list)
            keypoints_list_list.append(keypoints_list)
        else:
            #### 持续跟踪，当img_id是gt帧的时候会将gt和预测的进行比较.
            logger.info("Tracing,img_id:{}".format(img_id))
            candidates_total = []
            st_time_DET = time.time()
            candidates_from_detector = inference_yolov3(img_path)
            end_time_DET = time.time()
            total_time_DET += (end_time_DET - st_time_DET)

            candidates_from_prev = []

            bbox_list_prev_frame = []
            ''' 根据先前帧的信息补充框 '''
            if img_id > first_img_id:
                bbox_list_prev_frame = bbox_dets_list_list[prev_frame_img_id].copy()
                keypoints_list_prev_frame = keypoints_list_list[prev_frame_img_id].copy()
                num_prev_bbox = len(bbox_list_prev_frame)
                for prev_det_id in range(num_prev_bbox):
                    # obtain bbox position and track id
                    keypoints = keypoints_list_prev_frame[prev_det_id]['keypoints']
                    bbox_det_next = get_bbox_from_keypoints(keypoints)
                    if bbox_invalid(bbox_det_next):
                        continue
                    # xywh
                    candidates_from_prev.append(bbox_det_next)

            ''' 拿到本帧全部的候选框 '''
            candidates_total = candidates_from_detector + candidates_from_prev
            num_candidate = len(candidates_total)
            ''' 使用关节点的置信度来作为bbox的置信度 '''
            candidates_dets = []
            for candidate_id in range(num_candidate):
                bbox_det = candidates_total[candidate_id]
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": candidate_id,
                                 "imgpath": img_path,
                                 "track_id": None,
                                 "bbox": bbox_det}
                st_time_pose = time.time()
                keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]['keypoints']
                end_time_pose = time.time()
                total_time_POSE_ESTIMATOR += (end_time_pose - st_time_pose)
                bbox_det_next = xywh_to_x1y1x2y2(bbox_det)
                score = sum(keypoints[2::3]) / keypoints_number
                # 不知道为什么他这个pose的置信度会高于1
                if bbox_invalid(bbox_det_next) or score < 0.7:
                    filter_bbox_number += 1
                    continue
                candidate_det = bbox_det_next + [score]
                candidates_dets.append(candidate_det)
                keypoints_dict = {"img_id": img_id,
                                  "det_id": candidate_id,
                                  "imgpath": img_path,
                                  "track_id": None,
                                  "keypoints": keypoints}

                bbox_dets_list.append(bbox_det_dict)
                keypoints_list.append(keypoints_dict)
            # 根据bbox的置信度来使用nms
            keep = py_cpu_nms(np.array(candidates_dets, dtype=np.float32), 0.5) if len(candidates_dets) > 0 else []

            candidates_total = np.array(candidates_total)[keep]
            t = bbox_dets_list.copy()
            k = keypoints_list.copy()
            # 筛选过后的
            bbox_dets_list = [t[i] for i in keep]
            keypoints_list = [k[i] for i in keep]
            """ Data association """
            cur_det_number = len(candidates_total)
            prev_det_number = len(bbox_list_prev_frame)
            if img_id == first_img_id or prev_det_number == 0:
                for det_id, bbox_det_dict in enumerate(bbox_dets_list):
                    keypoints_dict = keypoints_list[det_id]
                    bbox_det_dict['det_id'] = det_id
                    keypoints_dict['det_id'] = det_id
                    track_id = next_id
                    bbox_det_dict['track_id'] = track_id
                    keypoints_dict['track_id'] = track_id
                    next_id = max(next_id, track_id)
                    next_id += 1
            else:
                scores = np.zeros((cur_det_number, prev_det_number))
                for det_id in range(cur_det_number):
                    bbox_det_dict = bbox_dets_list[det_id]
                    keypoints_dict = keypoints_list[det_id]
                    bbox_det = bbox_det_dict['bbox']
                    keypoints = keypoints_dict['keypoints']

                    # 计算当前帧的bbox和先前帧bboxes的分数
                    for prev_det_id in range(prev_det_number):
                        prev_bbox_det_dict = bbox_list_prev_frame[prev_det_id]
                        prev_keypoints_dict = keypoints_list_prev_frame[prev_det_id]
                        iou_score = iou(bbox_det, prev_bbox_det_dict['bbox'], xyxy=False)
                        if iou_score > 0.5:
                            scores[det_id, prev_det_id] = iou_alpha1 * iou_score

                st_time_ass = time.time()
                bbox_dets_list, keypoints_list, now_next_id = bipartite_graph_matching(bbox_dets_list,
                                                                                       bbox_list_prev_frame, scores,
                                                                                       keypoints_list, next_id)
                end_time_ass = time.time()
                total_time_ASSOCIATE += (end_time_ass - st_time_ass)

                next_id = now_next_id

            if len(bbox_dets_list) == 0:
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": 0,
                                 "track_id": None,
                                 "imgpath": img_path,
                                 "bbox": [0, 0, 2, 2]}
                bbox_dets_list.append(bbox_det_dict)

                keypoints_dict = {"img_id": img_id,
                                  "det_id": 0,
                                  "track_id": None,
                                  "imgpath": img_path,
                                  "keypoints": []}
                keypoints_list.append(keypoints_dict)

            bbox_dets_list_list.append(bbox_dets_list)
            keypoints_list_list.append(keypoints_list)
            ##########################################
            #### 如果是gt帧则会与预测帧的结果进行比较 ####
            ##########################################
            if img_id in gt_frame_index_list and gt_frame_index_list.index(img_id) >= 1:
                logger.info("type:{},img_id:{}".format('gt_guide', img_id))
                # gt frame
                num_dets = len(candidates_info)

                bbox_list_prediction = bbox_dets_list_list[img_id - first_img_id].copy()
                keypoints_list_prediction = keypoints_list_list[img_id - first_img_id].copy()
                bbox_list_gt = []
                keypoints_list_gt = []
                for det_id in range(num_dets):  # 组装gt帧的候选信息。
                    track_id, bbox_det, keypoints = get_candidate_info_opensvai_json(candidates_info, det_id)
                    bbox_det_dict = {"img_id": img_id,
                                     "det_id": det_id,
                                     "imgpath": img_path,
                                     "track_id": track_id,
                                     "bbox": bbox_det}
                    keypoints_dict = {"img_id": img_id,
                                      "det_id": det_id,
                                      "imgpath": img_path,
                                      "track_id": track_id,
                                      "keypoints": keypoints}

                    bbox_list_gt.append(bbox_det_dict)
                    keypoints_list_gt.append(keypoints_dict)
                if len(bbox_list_gt) > 0:
                    # 如果gt帧是有bbox的，那么使用gt的信息覆盖.  覆盖前需要纠正gt帧的track_id 。
                    bbox_dets_list_list[img_id - first_img_id] = bbox_list_gt
                    keypoints_list_list[img_id - first_img_id] = keypoints_list_gt
                    ## TODO 纠正gt的track_id
                ### 判断是否需要去纠正gt的前几帧。
                need_correct = distance_between_gt_prediction(
                    gt_dict={"det": bbox_list_gt, "keypoints": keypoints_list_gt},
                    predict_dict={"det": bbox_list_prediction,
                                  "keypoints": keypoints_list_prediction})
                if need_correct:
                    ## 往前进行矫正
                    correct_index = img_id - 1
                    correct_end_index = img_id - int(interval / 2)
                    # 从后往前
                    while correct_index >= correct_end_index:
                        ## 假设框是对的，id错了
                        ## 此时的prev_det_number 是gt
                        bbox_dets_list = bbox_dets_list_list[correct_index - first_img_id]
                        keypoints_list = keypoints_list_list[correct_index - first_img_id]

                        cur_det_number = len(bbox_dets_list)
                        gt_det_number = len(bbox_list_gt)
                        # gt 是已完成匹配的，cur是待匹配的
                        scores = np.zeros((cur_det_number, gt_det_number))
                        for cur_det_id in range(cur_det_number):
                            bbox_det_dict = bbox_dets_list[cur_det_id]
                            keypoints_dict = keypoints_list[cur_det_id]
                            bbox_det = bbox_det_dict['bbox']
                            keypoints = keypoints_dict['keypoints']

                            # 计算当前帧的bbox和先前帧bboxes的分数,
                            for gt_det_id in range(gt_det_number):
                                bbox_det_dict_gt = bbox_list_gt[gt_det_id]
                                iou_score = iou(bbox_det, bbox_det_dict_gt['bbox'], xyxy=False)
                                if iou_score > 0.2:
                                    scores[cur_det_id, gt_det_id] = iou_alpha1 * iou_score

                        if gt_det_number > 0 and cur_det_number > 0:
                            bbox_dets_list, keypoints_list, now_next_id = bipartite_graph_matching(bbox_dets_list,
                                                                                                   bbox_list_gt,
                                                                                                   scores,
                                                                                                   keypoints_list,
                                                                                                   next_id)

                        # 这一帧没有一个保留下来的bbox
                        if len(bbox_dets_list) == 0:
                            bbox_det_dict = {"img_id": img_id,
                                             "det_id": 0,
                                             "track_id": None,
                                             "imgpath": img_path,
                                             "bbox": [0, 0, 2, 2]}
                            bbox_dets_list.append(bbox_det_dict)

                            keypoints_dict = {"img_id": img_id,
                                              "det_id": 0,
                                              "track_id": None,
                                              "imgpath": img_path,
                                              "keypoints": []}
                            keypoints_list.append(keypoints_dict)

                        bbox_dets_list_list[correct_index - first_img_id] = bbox_dets_list.copy()
                        keypoints_list_list[correct_index - first_img_id] = keypoints_list.copy()
                        correct_index -= 1

        img_id += 1

    ''' 1. statistics: get total time for lighttrack processing'''
    end_time_total = time.time()
    total_time_ALL += (end_time_total - st_time_total)

    # convert results into openSVAI format
    print("Exporting Results in openSVAI Standard Json Format...")
    poses_standard = pose_to_standard_mot(keypoints_list_list, bbox_dets_list_list)
    # json_str = python_to_json(poses_standard)
    # print(json_str)

    # output json file
    pose_json_folder, _ = get_parent_folder_from_path(output_json_path)
    create_folder(pose_json_folder)
    write_json_to_file(poses_standard, output_json_path)
    print("Json Export Finished!")

    # visualization
    if flag_visualize is True:
        print("Visualizing Pose Tracking Results...")
        create_folder(visualize_folder)
        visualizer.show_all_from_standard_json(output_json_path, classes, joint_pairs, joint_names,
                                               image_folder,
                                               visualize_folder,
                                               flag_track=True)
        print("Visualization Finished!")

        img_paths = get_immediate_childfile_paths(visualize_folder)
        avg_fps = total_num_FRAMES / total_time_ALL
        # make_video_from_images(img_paths, output_video_path, fps=avg_fps, size=None, is_color=True, format="XVID")

        fps = 5  # 25 原来
        visualizer.make_video_from_images(img_paths, output_video_path, fps=fps, size=None, is_color=True,
                                          format="XVID")


def distance_between_gt_prediction(gt_dict, predict_dict):
    """
    判断是否需要矫正
    :param gt_dict:
    :param predict_dict:
    :return:
    """
    gt_det_list = gt_dict['det']
    gt_keypoints_list = gt_dict['keypoints']
    predict_det_list = predict_dict['det']
    predict_keypoints_list = predict_dict['keypoints']
    # TODO
    # for gt_det_id in gt_det_list:
    #     gt_det = gt_det_list[gt_det_id]
    #     gt_track_id = gt_det['track_id']
    #     for predict_det_id in predict_det_list:
    #         predict_det = predict_det_list[predict_det_id]
    #         predict_track_id = predict_det['track_id']
    # if predict_track_id == gt_track_id:

    return True


def find_gt_frame_index_list(gt_info, interval=5):
    gt_index_list = []
    prev_gt_index = -1
    for index in range(len(gt_info)):
        if gt_info[index]['labeled'] is True and (len(gt_index_list) == 0 or (index - prev_gt_index) % interval == 0):
            prev_gt_index = index
            gt_index_list.append(index)
    return gt_index_list


def bipartite_graph_matching(current_bbox_dict_list, prev_bbox_dict_list, score_between_two_frames,
                             current_keypoints_dict_list, next_id):
    """
        二分图的最大权值匹配
        current_bbox_dict_list ,不带track_id
        prev_bbox_dict_list    ,带track_id
        score_between_two_frames,  [current_bbox_id,prev_bbox_id]
    :param current_bbox_dict_list:
    :param prev_bbox_dict_list:
    :param score_between_two_frames:
    :param current_keypoints_dict_list:
    :param next_id:
    :return:
    """
    prev_to_cur_match = Kuhn_Munkras_match(current_bbox_dict_list, prev_bbox_dict_list, score_between_two_frames)
    result_bbox_dict_list = []
    result_keypoints_dict_list = []

    effective_prev_bbox_number = 0
    for prev_bbox_dict in prev_bbox_dict_list:
        if not bbox_invalid(prev_bbox_dict['bbox']):
            effective_prev_bbox_number = effective_prev_bbox_number + 1
    det_number = 0
    assigned_cur_bbox = []
    if effective_prev_bbox_number > 0:
        for prev_index, cur_index in enumerate(prev_to_cur_match):
            if not np.isnan(cur_index):
                assigned_cur_bbox.append(cur_index)
                cur_index = int(cur_index)
                cur_bbox_dict = current_bbox_dict_list[cur_index]
                cur_keypoints_dict = current_keypoints_dict_list[cur_index]
                prev_track_id = prev_bbox_dict_list[prev_index]['track_id']
                cur_bbox_dict['det_id'] = det_number
                cur_bbox_dict['track_id'] = prev_track_id
                cur_keypoints_dict['det_id'] = det_number
                cur_keypoints_dict['track_id'] = prev_track_id
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
    else:
        for current_index in range(len(current_bbox_dict_list)):
            cur_bbox_dict = current_bbox_dict_list[current_index]
            cur_keypoints_dict = current_keypoints_dict_list[current_index]
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


if __name__ == '__main__':

    global args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--video_path', '-v', type=str, dest='video_path',
    #                     # default="data/demo/video.mp4")
    #                     default="data/demo/0003.m4")
    # parser.add_argument('--images_path', '-i', type=str, dest='images_path',
    #                     default="data/demo/mpii-video-pose/0001")
    # parser.add_argument('--model', '-m', type=str, dest='test_model',
    #                     default="weights/mobile-deconv/snapshot_296.ckpt")
    # parser.add_argument('--model', '-m', type=str, dest='test_model',
    #                     default="weights/CPN101/CPN_snapshot_293.ckpt")
    parser.add_argument('--model', '-m', type=str, dest='test_model',
                        default="weights/MSRA152/MSRA_snapshot_285.ckpt")
    # default="weights/mobile-deconv/snapshot_296.ckpt")
    parser.add_argument('--train', type=bool, dest='train',
                        default=True)
    # parser.add_argument('--exp_number', type=str, dest='exp_number', default='2017-val',
    #                     help='number of experiment')
    parser.add_argument('--exp_number', type=str, dest='exp_number', default='test_one_video',
                        help='number of experiment')
    args = parser.parse_args()
    args.bbox_thresh = 0.4

    # initialize pose estimator
    initialize_parameters()
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(args.test_model)

    train = args.train
    exp_number = args.exp_number

    ##################################
    test_one_video = False
    # exp_number = "test-one-video-MSRA152-guide"
    val = True
    # exp_number = "2017-val-iou-{}-pose{}-together-MSRA152".format(iou_alpha1, pose_alpha1)
    version = 1
    exp_number = "2017-val-iou-{}-together-MSRA152-guide-v{}".format(iou_alpha1, version)
    test = False
    # exp_number = "2017-test-iou-pose-together"
    experiment_output_root = '/home/chm/experiment'
    visualize_root_folder = "{}/exp_{}/visualize".format(experiment_output_root, exp_number)
    output_video_folder = "{}/exp_{}/videos".format(experiment_output_root, exp_number)
    output_json_folder = "{}/exp_{}/jsons".format(experiment_output_root, exp_number)
    evaluation_folder = "{}/exp_{}/evaluation".format(experiment_output_root, exp_number)
    logger_file_foler = "{}/exp_{}/log".format(experiment_output_root, exp_number)

    create_folder(output_video_folder)
    create_folder(output_json_folder)
    create_folder(logger_file_foler)

    create_folder(evaluation_folder)
    create_folder(os.path.join(evaluation_folder, "annotations", "val"))
    create_folder(os.path.join(evaluation_folder, "out"))
    create_folder(os.path.join(evaluation_folder, "posetrack_results"))
    ## save log file
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG,
                        filename=os.path.join(logger_file_foler, 'experiment.log'),
                        filemode='a')
    ####################################

    logger.info(" test_one_video:{}  val:{}  test:{} ".format(test_one_video, val, test))

    """ 每个JSON文件为一个视频，读取一个个的JSON文件，产生结果 """
    if test_one_video:
        # numbers = ['24642', '24635', '23699', '23695', '23484', '23471', '23416', '22682', '22671', '22642', '22124',
        #            '00043', '00096']
        # numbers = ['24642', '24635', '23699', '23695', '23484']
        posetrack_dataset_path = "/home/chm/Dataset/PoseTrack2017/posetrack_data"
        numbers = ['20880']
        # input_jsons = ["/home/chm/Dataset/PoseTrack2017/posetrack_data/OpenSVAI_json/train/{}_mpii_relpath_5sec_trainsub_OpenSVAI.json".format(number)for number in numbers]
        input_jsons = [
            "/home/chm/Dataset/PoseTrack2017/posetrack_data/OpenSVAI_json/val/{}_mpii_relpath_5sec_testsub_OpenSVAI.json".format(
                number) for number in numbers]
        frame_number = 0
        videos_number = len(input_jsons)
        for input_json in tqdm(input_jsons):
            videos_json_data, videos_number = read_opensvai_json(input_json)
            for video_seq_id in range(videos_number):
                video_json_data = videos_json_data[video_seq_id]
                video_path, video_name = read_video_data_opensvai_json(video_json_data)
                image_folder = os.path.join(posetrack_dataset_path, video_path)
                visualize_folder = os.path.join(visualize_root_folder, video_name)
                output_video_path = os.path.join(output_video_folder, "{}_out.mp4".format(video_name))
                output_json_path = os.path.join(output_json_folder, "{}.json".format(video_name))
                create_folder(visualize_folder)
                frame_number += len(video_json_data)
                light_track(pose_estimator, image_folder, output_json_path, visualize_folder, output_video_path,
                            video_json_data)
        logger.info("videos_number:{}".format(videos_number))
        logger.info("frames_number:{}".format(frame_number))

    """ The PoseTrack2017 validation set """
    if val:
        # input_jsons_folder = "/media/D/DataSet/PoseTrack2017/val/"
        input_jsons_folder = "/home/chm/Dataset/PoseTrack2017/posetrack_data/OpenSVAI_json/val"
        posetrack_dataset_path = "/home/chm/Dataset/PoseTrack2017/posetrack_data"
        val_jsons = os.listdir(input_jsons_folder)
        frame_number = 0
        videos_number = len(val_jsons)
        for json in val_jsons:
            videos_json_data, _ = read_opensvai_json(os.path.join(input_jsons_folder, json))
            assert len(videos_json_data) == 1
            video_json_data = videos_json_data[0]
            video_path, video_name = read_video_data_opensvai_json(video_json_data)
            image_folder = os.path.join(posetrack_dataset_path, video_path)
            visualize_folder = os.path.join(visualize_root_folder, video_name)
            output_video_path = os.path.join(output_video_folder, "{}_out.mp4".format(video_name))
            output_json_path = os.path.join(output_json_folder, "{}.json".format(video_name))
            create_folder(visualize_folder)
            frame_number += len(video_json_data)
            light_track(pose_estimator, image_folder, output_json_path, visualize_folder, output_video_path,
                        video_json_data)

        logger.info("videos_number:{}".format(videos_number))
        logger.info("frames_number:{}".format(frame_number))
    """ The PoseTrack2017 test set """
    if test:
        input_jsons_folder = "/media/D/DataSet/PoseTrack2017/test/"
        posetrack_dataset_path = "/media/D/DataSet/PoseTrack/PoseTrack2017/posetrack_data"
        val_jsons = os.listdir(input_jsons_folder)
        frame_number = 0
        videos_number = len(val_jsons)
        for json in val_jsons:
            videos_json_data, _ = read_opensvai_json(os.path.join(input_jsons_folder, json))
            assert len(videos_json_data) == 1
            video_json_data = videos_json_data['annolist']
            video_path, video_name = read_video_data_opensvai_json(video_json_data)
            image_folder = os.path.join(posetrack_dataset_path, video_path)
            visualize_folder = os.path.join(visualize_root_folder, video_name)
            output_video_path = os.path.join(output_video_folder, "{}_out.mp4".format(video_name))
            output_json_path = os.path.join(output_json_folder, "{}.json".format(video_name))
            create_folder(visualize_folder)
            frame_number += len(video_json_data)
            light_track(pose_estimator, image_folder, output_json_path, visualize_folder, output_video_path,
                        video_json_data)

        logger.info("videos_number:{}".format(videos_number))
        logger.info("frames_number:{}".format(frame_number))

    ''' Display statistics '''
    logger.info("total_time_ALL: {:.2f}s".format(total_time_ALL))
    logger.info("total_time_DET: {:.2f}s".format(total_time_DET))
    logger.info("total_time_POSE_ESTIMATOR: {:.2f}s".format(total_time_POSE_ESTIMATOR))
    logger.info("total_time_POSE_SIMILARITY: {:.2f}s".format(total_time_POSE_SIMILARITY))
    logger.info("total_time_ASSOCIATE: {:.2f}s".format(total_time_ASSOCIATE))
    logger.info("total_time_LIGHTTRACK: {:.2f}s".format(
        total_time_ALL - total_time_DET - total_time_POSE_ESTIMATOR - total_time_POSE_SIMILARITY - total_time_ASSOCIATE))
    logger.info("filter_bbox_number:{}".format(filter_bbox_number))
