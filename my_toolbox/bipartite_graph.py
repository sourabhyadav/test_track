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


def Kuhn_Munkras_match(current_bboxes, prev_bboxes, scores):
    """
    Kuhn_Munkras algorithm
    :param current_bboxes:
    :param prev_bboxes:
    :param scores :  分数， type:np.array    , shape: [cur_number,prev_number]
    :return: 返回prev_bboxes的匹配列表;如果值为nan，则表示没有找到匹配
    """
    scores = np.where(scores > 0.1, scores, 0)

    cur_number = len(current_bboxes)
    prev_number = len(prev_bboxes)
    vis_cur = np.full(cur_number, False)
    vis_prev = np.full(prev_number, False)

    # 将关联边的最大权值赋为初值
    expected_cur = np.max(scores, axis=1)
    expected_prev = np.zeros(prev_number)

    match = np.full(prev_number, np.nan)
    slacks = np.full(prev_number, np.inf)

    # 为当前帧的每个bbox寻找在 previous帧的bbox
    for cur_id in range(cur_number):
        slacks = np.full(prev_number, np.inf)
        while True:

            vis_cur = np.full(cur_number, False)
            vis_prev = np.full(prev_number, False)

            # 成功找到了增广路，则该点增广，进入下一个点的增广
            if find_path_dfs(cur_id, vis_cur, vis_prev, expected_cur, expected_prev, slacks, scores,
                             match) or not np.isnan(match).any():
                break
            # 增广路寻找失败：则需要改变顶标，使图中可行边的数量增加。
            # 方法：将所有在增广路（）的X方点的标号全部减去一个常数d，在增广路的Y方点的顶标加上一个常数d
            d = np.inf
            for index, slack in enumerate(slacks):
                if not vis_prev[index] and slack < d:
                    d = slack
            for k in range(cur_number):
                if vis_cur[k]:
                    expected_cur[k] -= d
            for n in range(prev_number):
                if vis_prev[n]:
                    expected_prev[n] += d

    return match


def find_path_dfs(cur_id, vis_cur, vis_prev, expected_cur, expected_prev, slacks, scores, match):
    """
        find path deep first
    :return:
    """
    vis_cur[cur_id] = True
    for prev_index, match_score in enumerate(scores[cur_id]):
        if vis_prev[prev_index]:
            # 已被匹配
            continue
        gap = expected_cur[int(cur_id)] + expected_prev[int(prev_index)] - match_score
        if gap <= 0:
            vis_prev[prev_index] = True
            # prev_index 未被匹配，或者
            if np.isnan(match[prev_index]) or find_path_dfs(int(match[prev_index]), vis_cur, vis_prev, expected_cur,
                                                            expected_prev, slacks, scores, match):
                match[prev_index] = cur_id
                return True
        # 不在相等子图中slack取最小的
        elif slacks[prev_index] > gap:
            slacks[prev_index] = gap
    return False


def find_path_bfs():
    """
        find path breadth first
    :return:
    """


if __name__ == '__main__':
    print("123")
    current_bboxes = [0, 1, 2]
    prev_bboxes = [0, 1, 2, 3]
    scores = np.array([[0.234, 0.654, 0, 0.32345],
                       [0.123123, 0.5654, 0.1876, 0],
                       [0, 0.66456, 0.4423, 0.5123]])
    res = Kuhn_Munkras_match(current_bboxes, prev_bboxes, scores)
    print("asd")
    print("res:{}".format(res))
