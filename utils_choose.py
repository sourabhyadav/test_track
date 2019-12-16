import numpy as np
import torch


def select_bbox_by_criterion(bbox_dets_list, keypoints_list, confidence_scores, pose_matching_scores, iou_scores,
                             prev_track_ids, not_filter=False):
    '''
    根据三个得分来选择
    :param candidate_bbox:  candidate_bbox 经过本函数会变得
    :param confidence_scores:  [cur_bbox_number,confidence]
    :param pose_matching_scores: [cur_bbox_number,pose_match_score_to_prev]
        姿态匹配值，值越小越匹配。 range [0,1]
    :param iou_scores: iou
        并交比 值越大表示越相似
    :return: 返回一个numpy.array,长度一般来说等于prev frame 的bbox个数。
        [pre_bbox1_in_cur_frame_bbox_index,pre_bbox2_in_cur_frame_bbox_index,....,pre_bboxN_in_cur_frame_bbox_index]
    '''
    assert confidence_scores.shape[0] == pose_matching_scores.shape[0] == iou_scores.shape[0] and \
           pose_matching_scores.shape[1] == \
           iou_scores.shape[1]
    ### 仅用于代码测试阶段
    if not_filter:
        for index in range(len(bbox_dets_list)):
            bbox_dets_list[index]['track_id'] = index
            bbox_dets_list[index]['det_id'] = index

            keypoints_list[index]['track_id'] = index
            keypoints_list[index]['det_id'] = index
        return bbox_dets_list, keypoints_list

    cur_bbox_number, prev_bbox_number = pose_matching_scores.shape
    scale_confidence = 1
    scale_pose = -1
    scale_iou = 1
    # total_score = np.zeros_like[pose_matching_scores.shape]
    # [cur_box_index,prev_box]
    total_score = scale_confidence * confidence_scores + scale_pose * pose_matching_scores + scale_iou * iou_scores

    # max_index = np.argmax(total_score, axis=0)  # 直接找最大的下标
    # total_score ()

    dets = np.array([bbox_dets_list[det_id]['bbox'] for det_id in range(len(bbox_dets_list))])
    cnt = 0
    temp_bbox_det_list = []
    temp_keypoint_det_list = []
    invaild_flag = -2

    # 如果其 pose match score 大于阈值，则设设置为无效
    # 如果其 iou score 小于阈值，则设置为无效
    # total_score = np.where(pose_matching_scores < 0.4, total_score,invaild_flag) # 不一定可靠
    total_score = np.where(iou_scores > 0.5, total_score,invaild_flag)

    while cnt < prev_bbox_number and np.max(total_score) != invaild_flag:
        argmax = np.argmax(total_score)  # 按行优先拿到下标
        row_id = int(argmax % prev_bbox_number)  # row-列
        cur_index = int(argmax / prev_bbox_number)  # col-行

        track_id = prev_track_ids[row_id]

        bbox_dets_list[cur_index]['track_id'] = track_id
        bbox_dets_list[cur_index]['det_id'] = cnt

        keypoints_list[cur_index]['track_id'] = track_id
        keypoints_list[cur_index]['det_id'] = cnt

        temp_bbox_det_list.append(bbox_dets_list[cur_index])
        temp_keypoint_det_list.append(keypoints_list[cur_index])
        total_score[cur_index, :] = invaild_flag
        total_score[:, row_id] = invaild_flag
        # bbox_det_dict = {'target_id': target_id,
        #                  'bbox': dets[cur_index]}
        # 要保留的
        cnt += 1

        # if max_index[row_index] == None:
        #     # 在本次detection中到了最符合的
        #     max_index[row_index] = col_index
        #     # 用它做一次nms
        #     total_score[col_index][:] = -10000
        #     cnt += 1
        # else:
        #     total_score[col_index][row_index] = -10000
        #     continue

    # target_lost_thread = 4

    return temp_bbox_det_list, temp_keypoint_det_list


##########################################
### copy from /detector/detector_utils ###
##########################################
#
# ## 针对单个框的nms
# def non_max_suppression(bbox_info, target_bbox_index, nms_thres=0.6):
#     """
#     Removes detections with lower object confidence score than 'conf_thres' and performs
#     Non-Maximum Suppression to further filter detections.
#     Returns detections with shape:
#         (x1, y1, x2, y2, object_conf, class_score, class_pred)
#     """
#     # prediction  [image_number,]
#     # From (center x, center y, width, height) to (x1, y1, x2, y2)
#     bbox_info[..., :4] = xywh2xyxy(bbox_info[..., :4])  # 前四位
#     # 所有的候选跟置信度最大的比较（也会和它自己比较）
#     large_overlap = bbox_iou(bbox_info[target_bbox_index, :4].unsqueeze(0), bbox_info[:, :4]) > nms_thres
#     # Indices of boxes with lower confidence scores, large IOUs and matching labels
#     invalid = large_overlap
#     # Merge overlapping bboxes by order of confidence
#     # bbox_info[0, :4] = (weights * bbox_info[invalid, :4]).sum(0) / weights.sum()
#     output = []
#     for index in range(len(large_overlap)):
#         if large_overlap[index] == False:
#             # 如果不相似则保留
#             output.append(index)
#     return output


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  #
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
