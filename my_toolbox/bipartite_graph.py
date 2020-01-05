#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/01/04
    Description:  match the two bbox between current frame and previous frame
"""
import numpy as np


def Hungarian_match(current_bboxes, prev_bboxes):
    """ Hungarian algorithm
      :param current_bboxes
            type : dict   ['bbox']=bbox_info (list of 4 length , x y w h)
      :param prev_bboxes
            type: dict
    """
    pass


def Kuhn_Munkras_match(current_bboxes, prev_bboxes):
    """
    Kuhn_Munkras algorithm
    :param current_bboxes:
    :param prev_bboxes:
    :return:
    """

    # 声明数据结构
    adj_matrix = build_graph()  # np array with dimension N*N

    # 初始化顶标
    label_left = np.max(adj_matrix, axis=1)  # init label for the left set
    label_right = np.zeros(N)  # init label for the right set

    # 初始化匹配结果
    match_right = np.empty(N) * np.nan

    # 初始化辅助变量
    visit_left = np.empty(N) * False
    visit_right = np.empty(N) * False
    slack_right = np.empty(N) * np.inf



int love[MAXN][MAXN];   // 记录每个妹子和每个男生的好感度
int ex_girl[MAXN];      // 每个妹子的期望值
int ex_boy[MAXN];       // 每个男生的期望值
bool vis_girl[MAXN];    // 记录每一轮匹配匹配过的女生
bool vis_boy[MAXN];     // 记录每一轮匹配匹配过的男生
int match[MAXN];        // 记录每个男生匹配到的妹子 如果没有则为-1
int slack[MAXN];        // 记录每个汉子如果能被妹子倾心最少还需要多少期望值

girl_number=10
boy_number=10

ex_girl= np.zeros([10])
ex_girl= np.zeros([10])

def find_path_dfs():
    """
        find path deep first
    :return:
    """

def find_path_bfs():
    """
        find path breadth first
    :return:
    """


def find_path(i):
    visit_left[i] = True
    for j, match_weight in enumerate(adj_matrix[i]):
        if visit_right[j]: continue  # 已被匹配（解决递归中的冲突）
        gap = label_left[i] + label_right[j] - match_weight
        if gap == 0:
            # 找到可行匹配
            visit_right[j] = True
            if np.isnan(match_right[j]) or find_path(match_right[j]):  ## j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选备胎
                match_right[j] = i
                return True
            else:
        # 计算变为可行匹配需要的顶标改变量
        if slack_right[j] < gap: slack_right[j] = gap
    return False
