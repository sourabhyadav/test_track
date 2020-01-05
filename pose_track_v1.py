#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2019/12/18
"""
import time
import argparse

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
import logging
# import Network
from network_mobile_deconv import Network

# detector utils
from detector.detector_yolov3 import *  ##

# pose estimation utils
from HPE.dataset import Preprocessing
from HPE.config import cfg
from tfflat.base import Tester
from tfflat.utils import mem_info
from tfflat.logger import colorlogger
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

# import GCN utils
from graph import visualize_pose_matching
from graph.visualize_pose_matching import *

# import my own utils
import sys, os, time

sys.path.append(os.path.abspath("./graph"))
sys.path.append(os.path.abspath("./utils"))

# print("sys.path", sys.path)
from utils_json import *
from utils_io_file import *
from utils_io_folder import *

# from .utils.utils_json import *

from visualizer import *
from visualizer import visualizer

from utils_choose import *

# from visualizer import visualizer

# from .utils.utils_io_file import *
# from .utils.utils_io_folder import *

flag_visualize = True
flag_nms = False  # Default is False, unless you know what you are doing

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################
##单纯为了Debug
image_crop_output_path = '/media/D/light-track/data/demo/crop'
image_seed_crop_output_path = '/media/D/light-track/data/demo/seed_crop'
tracking_gt_info = []

import logging
from sheen import Str, ColoredHandler

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.addHandler(ColoredHandler())


################

def initialize_parameters():
    global video_name, img_id

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

    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    total_time_POSE = 0
    total_time_DET = 0
    total_time_ALL = 0
    total_num_FRAMES = 0
    total_num_PERSONS = 0
    return


def light_track(pose_estimator,
                image_folder, output_json_path,
                visualize_folder, output_video_path):
    global total_time_POSE, total_time_DET, total_time_ALL, total_num_FRAMES, total_num_PERSONS
    global video_name
    ''' 1. statistics: get total time for lighttrack processing'''
    st_time_total = time.time()

    # process the frames sequentially
    keypoints_list = []
    bbox_dets_list = []
    # frame_prev = -1
    # frame_cur = 0
    img_id = -1
    next_id = 0
    bbox_dets_list_list = []
    keypoints_list_list = []

    flag_mandatory_keyframe = False

    img_paths = get_immediate_childfile_paths(image_folder)
    num_imgs = len(img_paths)
    total_num_FRAMES = num_imgs

    # 有gt的的bbox
    gt_bbox_img_id_list = [0]

    seed_mode = False

    while img_id < num_imgs - 1:
        img_id += 1
        img_path = img_paths[img_id]
        print("Current tracking: [image_id:{}]".format(img_id))
        frame_cur = img_id

        bbox_dets_list = []  # keyframe: start from empty
        keypoints_list = []  # keyframe: start from empty

        if img_id in gt_bbox_img_id_list:
            # 当前帧是gt帧
            # 当做好数据处理后，要用gt来做，现在是伪gt
            ##  TODO 带数据弄好后 remove
            human_candidates = inference_yolov3(img_path)  # 拿到bbox
            num_dets = len(human_candidates)
            # 检测bbox的keypoints
            for det_id in range(num_dets):
                bbox_det = human_candidates[det_id]
                bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
                bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
                # update current frame bbox
                bbox_det_dict = {"img_id": img_id,
                                 "det_id": det_id,
                                 "imgpath": img_path,
                                 "track_id": det_id,
                                 "bbox": bbox_det}

                # keypoint检测，并记录时间
                st_time_pose = time.time()
                keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]
                end_time_pose = time.time()
                total_time_POSE += (end_time_pose - st_time_pose)

                keypoints_dict = {"img_id": img_id,
                                  "det_id": det_id,
                                  "imgpath": img_path,
                                  "track_id": det_id,
                                  "keypoints": keypoints}
                bbox_dets_list.append(bbox_det_dict)
                keypoints_list.append(keypoints_dict)
            # assert len(bbox_dets_list) == 2
            bbox_dets_list_list.append(bbox_dets_list)
            keypoints_list_list.append(keypoints_list)
        else:
            # 当前帧非gt帧
            # perform detection at keyframes
            if seed_mode:
                logger.info("img_id:{},seed_mode".format(img_id))
                # 拿到上一帧的信息
                bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()
                keypoints_list_prev_frame = keypoints_list_list[img_id - 1].copy()
                num_prev_bbox = len(bbox_list_prev_frame)

                my_enlarge_scale = 0.3
                cur_image = cv2.imread(img_path)
                cur_image_name = os.path.basename(img_path).split('.')[0]
                cnt = 0
                for prev_det_id in range(num_prev_bbox):
                    prev_bbox_det = bbox_list_prev_frame[prev_det_id]["bbox"]  # xywh
                    track_id = bbox_list_prev_frame[prev_det_id]['track_id']
                    prev_enlarge_bbox_det = x1y1x2y2_to_xywh(
                        enlarge_bbox(xywh_to_x1y1x2y2(prev_bbox_det), my_enlarge_scale))
                    x1, x2, y1, y2 = max(0, int(prev_enlarge_bbox_det[0])), int(
                        prev_enlarge_bbox_det[0] + prev_enlarge_bbox_det[2]), \
                                     max(0, int(prev_enlarge_bbox_det[1])), int(
                        prev_enlarge_bbox_det[1] + prev_enlarge_bbox_det[3])
                    crop_image = cur_image[y1:y2, x1:x2].copy()
                    crop_image_folder_path = os.path.join(image_seed_crop_output_path, video_name, cur_image_name)
                    create_folder(crop_image_folder_path)
                    crop_image_path = os.path.join(crop_image_folder_path, "{:0>3d}".format(prev_det_id)) + '.jpg'
                    cv2.imwrite(crop_image_path, crop_image)
                    # 查看裁剪后的图片
                    human_candidates, confidence_scores = inference_yolov3_v1(crop_image_path)
                    logger.info(confidence_scores)
                    if len(human_candidates) > 0 and confidence_scores[0] > 0.90:
                        selected_bbox = human_candidates[0]
                        x1y1x2y2 = xywh_to_x1y1x2y2(selected_bbox)
                        # 左上角坐标
                        top_left_point_x, top_left_point_y = min(x1y1x2y2[0], x1y1x2y2[2]), min(x1y1x2y2[1],
                                                                                                x1y1x2y2[3])
                        best_bbox_det = [x1 + top_left_point_x, y1 + top_left_point_y, selected_bbox[2],
                                         selected_bbox[3]]

                        bbox_det_dict = {"img_id": img_id,
                                         "det_id": cnt,
                                         "imgpath": img_path,
                                         "track_id": track_id,
                                         "bbox": best_bbox_det}
                        crop_keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]
                        keypoints_dict = {"img_id": img_id,
                                          "det_id": cnt,
                                          "imgpath": img_path,
                                          "track_id": track_id,
                                          "keypoints": crop_keypoints}
                        bbox_dets_list.append(bbox_det_dict)
                        keypoints_list.append(keypoints_dict)
                        cnt += 1
                        # for proposal_det_id in range(num_proposal_dets):
                        #     proposal_bbox_det = human_candidates[proposal_det_id]
                        #     proposal_bbox_det_dict = {"img_id": 1,
                        #                               "imgpath": crop_image_path, "bbox": proposal_bbox_det}
                        #     crop_keypoints = inference_keypoints(pose_estimator, proposal_bbox_det_dict)[0][
                        #         "keypoints"]  # keypoint_numer *(x,y,score)
                        #     keypoint_sum_score = 0
                        #     for i in range(len(crop_keypoints)):
                        #         if i % 3 == 2:
                        #             keypoint_sum_score = keypoint_sum_score + crop_keypoints[i]
                        #     logger.info("{},{}".format(proposal_det_id, keypoint_sum_score))
                        #
                        #     crop_bbox_image_path = os.path.join(crop_image_folder_path,
                        #                                         "{:0>3d}-{:0>3d}".format(prev_det_id,
                        #                                                                  proposal_det_id)) + '.jpg'
                        #     cv2.imwrite(crop_bbox_image_path, cropped_bbox_image)
                assert cnt == len(bbox_dets_list)
                print("Final save bbox number: {} ".format(len(bbox_dets_list)))
                print("image path:{}".format(img_path))
                bbox_dets_list_list.append(bbox_dets_list)
                keypoints_list_list.append(keypoints_list)
                seed_mode = False
            else:
                st_time_detection = time.time()
                # human_candidates  ( center_x,center_y,w,h)
                human_candidates, confidence_scores = inference_yolov3_v1(img_path)  # 拿到bbox

                end_time_detection = time.time()
                total_time_DET += (end_time_detection - st_time_detection)

                num_dets = len(human_candidates)
                print("Keyframe: {} detections".format(num_dets))

                # if nothing detected at this frame
                if num_dets <= 0:
                    ## TODO
                    break

                # 检测bbox的keypoints
                for det_id in range(num_dets):
                    bbox_det = human_candidates[det_id]
                    bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                    bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, enlarge_scale)
                    bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
                    # update current frame bbox
                    bbox_det_dict = {"img_id": img_id,
                                     "det_id": det_id,
                                     "imgpath": img_path,
                                     "track_id": None,
                                     "bbox": bbox_det}

                    # keypoint检测，并记录时间
                    st_time_pose = time.time()
                    keypoints = inference_keypoints(pose_estimator, bbox_det_dict)[0]["keypoints"]
                    end_time_pose = time.time()
                    total_time_POSE += (end_time_pose - st_time_pose)

                    keypoints_dict = {"img_id": img_id,
                                      "det_id": det_id,
                                      "imgpath": img_path,
                                      "track_id": None,
                                      "keypoints": keypoints}
                    bbox_dets_list.append(bbox_det_dict)
                    keypoints_list.append(keypoints_dict)

                # 拿到上一帧的信息
                bbox_list_prev_frame = bbox_dets_list_list[img_id - 1].copy()
                keypoints_list_prev_frame = keypoints_list_list[img_id - 1].copy()

                ############ 裁剪
                # if img_id in [34, 35, 36, 37, 38]:
                #     cnt = 0
                #     for bbox_info in bbox_list_prev_frame:
                #         bbox_det = bbox_info['bbox']
                #         image_path = bbox_info['imgpath']
                #         frame_name = os.path.basename(image_path)
                #         frame_name = frame_name.split('.')[0]
                #         video_name = os.path.basename(image_folder)
                #         image = cv2.imread(image_path)
                #         bbox_x1y1x2y2 = xywh_to_x1y1x2y2(bbox_det)
                #         bbox_in_xywh = enlarge_bbox(bbox_x1y1x2y2, 0.1)
                #         bbox_det = x1y1x2y2_to_xywh(bbox_in_xywh)
                #         x1, y1, w, h = max(int(bbox_det[0]), 0), max(int(bbox_det[1]), 0), bbox_det[2], bbox_det[3]
                #         ### 得到裁剪后的图
                #         cropped_image = image[y1:(y1 + h), x1:(x1 + w)]
                #         create_folder(os.path.join(image_crop_output_path, video_name))
                #         cropped_image_path = os.path.join(image_crop_output_path, video_name,
                #                                           '{}-{:0>3d}.jpg'.format(frame_name, cnt))
                #         cv2.imwrite(cropped_image_path, cropped_image)
                #         ### 找bbox
                #         crop_human_candidates, _ = inference_yolov3_v1(cropped_image_path)
                #         for det_id in range(len(crop_human_candidates)):
                #             bbox_det = crop_human_candidates[det_id]
                #             ### 画bbox
                #             # cropped_bbox_image = visualizer.draw_bbox_from_python_data(cropped_image, bbox_det)
                #             cropped_bbox_image = cv2.rectangle(cropped_image.copy(), (int(bbox_det[0]), int(bbox_det[1])),
                #                                                (int(bbox_det[0] + bbox_det[2]),
                #                                                 int(bbox_det[1] + bbox_det[3])),
                #                                                (255, 0, 255), thickness=3)
                #             cropped_image_bbox_path = os.path.join(image_crop_output_path, video_name,
                #                                                    '{}-{:0>3d}-{:0>3d}.jpg'.format(frame_name, cnt, det_id))
                #             cv2.imwrite(cropped_image_bbox_path, cropped_bbox_image)
                #         cnt += 1

                ##############

                num_bbox_prev_frame = len(bbox_list_prev_frame)

                # 获取到三个指标的信息
                confidence_scores = np.array(confidence_scores)
                confidence_scores = confidence_scores[:, np.newaxis]
                pose_matching_scores = np.zeros([num_dets, num_bbox_prev_frame], dtype=float)
                iou_scores = np.ones([num_dets, num_bbox_prev_frame], dtype=float)
                prev_track_ids = []
                for bbox_prev_index in range(num_bbox_prev_frame):
                    # 上一帧中包含的trackIds
                    track_id = keypoints_list_prev_frame[bbox_prev_index]["track_id"]
                    prev_track_ids.append(track_id)
                for det_id in range(num_dets):
                    for bbox_prev_index in range(num_bbox_prev_frame):
                        keypoints_cur_frame = keypoints_list[det_id]["keypoints"]
                        bbox_cur_frame = bbox_dets_list[det_id]["bbox"]

                        keypoints_prev_frame = keypoints_list_prev_frame[bbox_prev_index]["keypoints"]
                        bbox_prev_frame = bbox_list_prev_frame[bbox_prev_index]["bbox"]
                        # get pose match score
                        pose_matching_scores[det_id, bbox_prev_index] = get_pose_matching_score(
                            keypoints_cur_frame,
                            keypoints_prev_frame,
                            bbox_cur_frame,
                            bbox_prev_frame)

                        # get bbox distance score
                        iou_scores[det_id, bbox_prev_index] = iou(bbox_cur_frame, bbox_prev_frame, xyxy=False)

                ###########################
                ## 根据指标来选择当前帧的框 ##
                ###########################
                bbox_dets_list, keypoints_list = select_bbox_by_criterion(bbox_dets_list, keypoints_list,
                                                                          confidence_scores,
                                                                          pose_matching_scores, iou_scores,
                                                                          prev_track_ids)
                num_save_bbox = len(bbox_dets_list)

                # 如果人数发生变化,该帧使用seed 模式
                if num_save_bbox < num_bbox_prev_frame:
                    seed_mode = True
                    img_id -= 1
                    continue
                print("Final save bbox number: {} ".format(len(bbox_dets_list)))
                print("image path:{}".format(img_path))
                bbox_dets_list_list.append(bbox_dets_list)
                keypoints_list_list.append(keypoints_list)

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


def get_iou_score(bbox_gt, bbox_det):
    boxA = xywh_to_x1y1x2y2(bbox_gt)
    boxB = xywh_to_x1y1x2y2(bbox_det)

    iou_score = iou(boxA, boxB)
    # print("iou_score: ", iou_score)
    return iou_score


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


def is_keyframe(img_id, interval=10):
    if img_id % interval == 0:
        return True
    else:
        return False


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


if __name__ == '__main__':

    global args
    ## from detector.detector_yolov3 import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', type=str, dest='video_path',
                        # default="data/demo/video.mp4")
                        default="data/demo/0003.m4")
    parser.add_argument('--images_path', '-i', type=str, dest='images_path',
                        default="data/demo/mpii-video-pose/0001")
    # default="data/demo/0002")
    parser.add_argument('--model', '-m', type=str, dest='test_model',
                        default="weights/mobile-deconv/snapshot_296.ckpt")

    args = parser.parse_args()
    args.bbox_thresh = 0.4

    # initialize pose estimator
    initialize_parameters()
    pose_estimator = Tester(Network(), cfg)
    pose_estimator.load_weights(args.test_model)

    video_path = args.video_path

    images_path = args.images_path

    visualize_folder = "data/demo/visualize/my"
    output_video_folder = "data/demo/videos/my"
    output_json_folder = "data/demo/jsons/my"

    ##
    # list_video = ['0004.mp4', '0005.mp4', '0006.mp4', '0007.mp4', '0008.mp4', '0009.mp4', '0010.mp4', '0011.mp4', '0012.mp4']
    # list_video_path = [os.path.join('data/demo', video) for video in list_video]
    #
    # for video_path_i in list_video_path:
    #     video_path = video_path_i
    #     video_to_images(video_path)
    #     video_name = os.path.basename(video_path)
    #     video_name = os.path.splitext(video_name)[0]
    #     image_folder = os.path.join("data/demo", video_name)
    #     visualize_folder = os.path.join(visualize_folder, video_name)
    #     output_json_path = os.path.join(output_json_folder, video_name + ".json")
    #     output_video_path = os.path.join(output_video_folder, video_name + "_out.mp4")
    #     create_folder(visualize_folder)
    #     create_folder(output_video_folder)
    #     create_folder(output_json_folder)
    #
    #     light_track(pose_estimator,
    #                 image_folder, output_json_path,
    #                 visualize_folder, output_video_path)
    list_video = ['PoseTrack2017/{0:06d}_bonn'.format(i) for i in [1, 3, 15, 17, 22, 26, 27, 28, 48]]
    for video in list_video:
        visualize_folder = "data/demo/visualize/my"
        output_video_folder = "data/demo/videos/my"
        output_json_folder = "data/demo/jsons/my"
        video_name = os.path.basename(video)
        image_folder = os.path.join("data/demo/PoseTrack2017", video_name)
        visualize_folder = os.path.join(visualize_folder, 'PoseTrack2017')
        output_json_folder = os.path.join(output_json_folder, 'PoseTrack2017')
        output_video_folder = os.path.join(output_video_folder, 'PoseTrack2017')

        visualize_folder = os.path.join(visualize_folder, video_name)
        output_json_path = os.path.join(output_json_folder, video_name + ".json")
        output_video_path = os.path.join(output_video_folder, video_name + "_out.mp4")
        create_folder(visualize_folder)
        create_folder(output_video_folder)
        create_folder(output_json_folder)

        light_track(pose_estimator,
                    image_folder, output_json_path,
                    visualize_folder, output_video_path)
    sys.exit()

    if is_video(video_path):
        video_to_images(video_path)
        video_name = os.path.basename(video_path)
        video_name = os.path.splitext(video_name)[0]
        image_folder = os.path.join("data/demo", video_name)
        visualize_folder = os.path.join(visualize_folder, video_name)
        output_json_path = os.path.join(output_json_folder, video_name + ".json")
        output_video_path = os.path.join(output_video_folder, video_name + "_out.mp4")
        create_folder(visualize_folder)
        create_folder(output_video_folder)
        create_folder(output_json_folder)

        light_track(pose_estimator,
                    image_folder, output_json_path,
                    visualize_folder, output_video_path)

        print("Finished video {}".format(output_video_path))

        ''' Display statistics '''
        print("total_time_ALL: {:.2f}s".format(total_time_ALL))
        print("total_time_DET: {:.2f}s".format(total_time_DET))
        print("total_time_POSE: {:.2f}s".format(total_time_POSE))
        print(
            "total_time_LIGHTTRACK: {:.2f}s".format(total_time_ALL - total_time_DET - total_time_POSE))
        print("total_num_FRAMES: {:d}".format(total_num_FRAMES))
        print("total_num_PERSONS: {:d}\n".format(total_num_PERSONS))
        print("Average FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
        print("Average FPS excluding Pose Estimation: {:.2f}fps".format(
            total_num_FRAMES / (total_time_ALL - total_time_POSE)))
        print("Average FPS excluding Detection: {:.2f}fps".format(
            total_num_FRAMES / (total_time_ALL - total_time_DET)))
        print("Average FPS for framework only: {:.2f}fps".format(
            total_num_FRAMES / (total_time_ALL - total_time_DET - total_time_POSE)))
    else:
        video_name = os.path.basename(images_path)
        image_folder = os.path.join("data/demo", video_name)
        visualize_folder = os.path.join(visualize_folder, video_name)
        output_json_path = os.path.join(output_json_folder, video_name + ".json")
        output_video_path = os.path.join(output_video_folder, video_name + "_out.mp4")
        create_folder(visualize_folder)
        create_folder(output_video_folder)
        create_folder(output_json_folder)

        light_track(pose_estimator,
                    image_folder, output_json_path,
                    visualize_folder, output_video_path)

        print("Finished video {}".format(output_video_path))

        ''' Display statistics '''
        print("total_time_ALL: {:.2f}s".format(total_time_ALL))
        print("total_time_DET: {:.2f}s".format(total_time_DET))
        print("total_time_POSE: {:.2f}s".format(total_time_POSE))
        print(
            "total_time_LIGHTTRACK: {:.2f}s".format(total_time_ALL - total_time_DET - total_time_POSE))
        print("total_num_FRAMES: {:d}".format(total_num_FRAMES))
        print("total_num_PERSONS: {:d}\n".format(total_num_PERSONS))
        print("Average FPS: {:.2f}fps".format(total_num_FRAMES / total_time_ALL))
        print("Average FPS excluding Pose Estimation: {:.2f}fps".format(
            total_num_FRAMES / (total_time_ALL - total_time_POSE)))
        print("Average FPS excluding Detection: {:.2f}fps".format(
            total_num_FRAMES / (total_time_ALL - total_time_DET)))
        print("Average FPS for framework only: {:.2f}fps".format(
            total_num_FRAMES / (total_time_ALL - total_time_DET - total_time_POSE)))
        # print("Video does not exist.")
