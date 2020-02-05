# #!/usr/bin/python
# # -*- coding:utf8 -*-
# """
#     Author: Haoming Chen
#     E-mail: chenhaomingbob@163.com
#     Time: 2020/01/06
#     Description:
# """
#
# import numpy as np
#
# # 声明数据结构
# adj_matrix = build_graph()  # np array with dimension N*N
#
# # 初始化顶标
# label_left = np.max(adj_matrix, axis=1)  # init label for the left set
# label_right = np.zeros(N)  # init label for the right set
#
# # 初始化匹配结果
# match_right = np.empty(N) * np.nan
#
# # 初始化辅助变量
# visit_left = np.empty(N) * False
# visit_right = np.empty(N) * False
# slack_right = np.empty(N) * np.inf
#
#
# # 寻找增广路，深度优先
# def find_path(i):
#     visit_left[i] = True
#     for j, match_weight in enumerate(adj_matrix[i]):
#         if visit_right[j]: continue  # 已被匹配（解决递归中的冲突）
#         gap = label_left[i] + label_right[j] - match_weight
#         if gap == 0:
#             # 找到可行匹配
#             visit_right[j] = True
#             if np.isnan(match_right[j]) or find_path(match_right[j]):  ## j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选备胎
#                 match_right[j] = i
#                 return True
#             else:
#         # 计算变为可行匹配需要的顶标改变量
#         if slack_right[j] < gap: slack_right[j] = gap
#     return False
#
#
# # KM主函数
# def KM():
#     for i in range(N):
#         # 重置辅助变量
#         slack_right = np.empty(N) * np.inf
#         while True:
#             # 重置辅助变量
#             visit_left = np.empty(N) * False
#             visit_right = np.empty(N) * False
#
#         # 能找到可行匹配
#         if find_path(i):    break
#         # 不能找到可行匹配，修改顶标
#         # (1)将所有在增广路中的X方点的label全部减去一个常数d
#         # (2)将所有在增广路中的Y方点的label全部加上一个常数d
#         d = np.inf
#         for j, slack in enumerate(slack_right):
#             if not visit_right[j] and slack < d:
#                 d = slack
#         for k in range(N):
#             if visit_left[k]: label_left[k] -= d
#         for n in range(N):
#             if visit_right[n]: label_right[n] += d
#
#
# res = 0
# for j in range(N):
#     if match_right[j] >= 0 and match_right[j] < N:
#         res += adj_matrix[match[j]][j]
# return res