#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2019/12/30
    Description:
"""
import os

## 评测前的预处理

gt_json_folder = '/media/F/exp_2017-val/evaluation/annotations/val/'
# input_json_folder = '/media/F/evaluation/Data_2017/posetrack_results/lighttrack/results_openSVAI'

dirs = os.listdir(gt_json_folder)
for json in dirs:
    prefix = json.split('_')[0:2]
    t = "{}_{}.json".format(prefix[0], prefix[1])
    os.rename(os.path.join(gt_json_folder, json), os.path.join(gt_json_folder, t))
    print(prefix)

# gt_json_folder = '/media/F/evaluation/Data_2017/posetrack_data/annotations/val'
# input_json_folder = '/media/F/evaluation/Data_2017/posetrack_results/lighttrack/results_openSVAI'
#
# dirs = os.listdir(gt_json_folder)
# cnt = 0
# for json in dirs:
#     prefix = json.split('_')[0:2]
#     t = "{}_{}.json".format(prefix[0], prefix[1])
#     file = os.path.join(input_json_folder, t)
#     print(file)
#     if os.path.isfile(file):
#         cnt += 1
#         os.rename(file, os.path.join(input_json_folder, json))
#     else:
#         os.remove(file)
#
# print(cnt)
# dirs = os.listdir(gt_json_folder)
#
# cnt = 0
# for json in dirs:
#     f = os.path.join(input_json_folder, json)
#     if not os.path.isfile(f):
#         cnt += 1
#         print(f)
#         # os.rename(file, os.path.join(input_json_folder, json))
#         os.remove(os.path.join(gt_json_folder, json))
# print(cnt)
