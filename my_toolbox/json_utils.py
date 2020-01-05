#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2019/12/24
    Description: jsond的数据参考 OpenSVAI

    OpenSVAI:http://openaccess.thecvf.com/content_ECCVW_2018/papers/11130/Ning_A_Top-down_Approach_to_Articulated_Human_Pose_Estimation_and_Tracking_ECCVW_2018_paper.pdf
"""
import json
import os


def read_opensvai_json(path):
    json_path = path
    data = json.loads(open(json_path, encoding='utf-8').read())
    video_number = len(data)

    return data, video_number


def read_video_data_opensvai_json(video_json_data):
    video_path = video_json_data[0]['image']['folder']
    video_name = os.path.basename(video_path)

    return video_path, video_name


def find_first_labeled_opensvai_json(video_json_data):
    for index in range(len(video_json_data)):
        if video_json_data[index]['labeled'] == True:
            return index
    return 0


def read_image_data_opensvai_json(image_json_data):
    image_name = image_json_data['image']['name']
    labeled = image_json_data['labeled']
    candidates_info = image_json_data['candidates']

    return image_name, labeled, candidates_info


def get_candidate_info_opensvai_json(candidates_info, index):
    assert index < len(candidates_info)
    det_bbox = candidates_info[index]['det_bbox']
    keypoints_2d = candidates_info[index]['pose_keypoints_2d']
    track_id = candidates_info[index]['track_id']
    return track_id, det_bbox, keypoints_2d


# test
if __name__ == '__main__':
    pass
