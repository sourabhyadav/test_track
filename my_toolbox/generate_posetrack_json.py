#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2019/12/18
"""

import os
from tqdm import tqdm
import json
from convert.general import *
from convert.box import *
from convert.loader import *
import glob
import collections
import shutil

DataSetInitArg = collections.namedtuple(
    'DataSetInitArg', ['mat_dir', 'out_path', 'recreate_videos', 'splits', 'vid_indir', 'vid_outdir']
)
ImageAnnot = collections.namedtuple(
    'ImageAnnot', [
        'folder',  # Path to the image
        'name',
        'id',  # start from 1
    ]
)
CandidatesAnnot = collections.namedtuple(
    'Candidate', [
        'track_score',  # range from 0 to 1
        'track_id',  # integer
        'det_score',  # range from 0 to 1
        'pose_keypoints_2d',  # np.array of 15x3 dimension
        'det_bbox'  # [min_x,min_y,w,h]
    ]
)
# mpii_video = DataSetInitArg(
#     mat_dir='/media/D/DataSet/PoseTrack/PoseTrack2017/posetrack_data/annotations/{}',
#     out_path='/media/D/DataSet/PoseTrack2017/posetrack_{}.json',
#     # splits=['train', 'val'],
#     recreate_videos=True,
#     vid_indir='/media/D/DataSet/PoseTrack/PoseTrack2017/posetrack_data/images',
#     vid_outdir='/media/D/DataSet/PoseTrack2017/images_renamed'
# )

PoseTrack = DataSetInitArg(
    mat_dir='/media/D/DataSet/PoseTrack/PoseTrack2017/posetrack_data/annotations/{}',
    out_path='/media/D/DataSet/PoseTrack2017/gt/posetrack_{}.json',
    splits=['train', 'val'],
    recreate_videos=False,
    vid_indir='/media/D/DataSet/PoseTrack/PoseTrack2017/posetrack_data/images',
    vid_outdir='/media/D/DataSet/PoseTrack2017/images_renamed'
)


def get_posetrack_kpt_ordering():
    # posetrack 有15个关键点
    ordering = [
        'right_ankle',
        'right_knee',
        'right_hip',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_wrist',
        'right_elbow',
        'right_shoulder',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'head_bottom',
        'nose',
        'head_top',
    ]
    joints = [
        ('head_top', 'nose'),
        ('nose', 'head_bottom'),
        ('head_bottom', 'left_shoulder'),
        ('head_bottom', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('head_bottom', 'left_hip'),
        ('head_bottom', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
    ]
    return ordering, joints


def _load_mat_files(annot_dir):
    mat_data = {}
    print('Loading data from MAT files...')
    # glob.glob 匹配所有的符合条件的文件，并将其以list的形式返回
    for fpath in tqdm(glob.glob(os.path.join(annot_dir, '*.mat'))):
        stuff = load_mat(fpath)
        if len(stuff) > 0:
            key = os.path.dirname(stuff[0].im_name)
            key = key[len('images/'):]
            mat_data[key] = stuff
    return mat_data


def _convert_video_frame_ids(inpath, outpath):
    """
    PoseTrack videos follow no consistent naming for frames. Make it consistent
    """
    mkdir_p(outpath)
    frame_names = [os.path.basename(el) for el in glob.glob(os.path.join(
        inpath, '*.jpg'))]
    # Some videos have 00XX_crop.jpg style filenames
    frame_ids = [int(el.split('.')[0].split('_')[0]) for el in frame_names]
    id_to_name = dict(zip(frame_ids, frame_names))
    for i, fid in enumerate(sorted(frame_ids)):
        shutil.copy('{}/{}'.format(inpath, id_to_name[fid]),
                    '{}/{:08d}.jpg'.format(outpath, i + 1))


def _convert_mat_to_posetrack_json(annot_dir, out_path, vid_indir, vid_outdir, recreate_videos):
    json_data = []
    ''' 分为train val 两个json'''
    for fpath in tqdm(glob.glob(os.path.join(annot_dir, '*.mat'))):
        pictures = load_mat(fpath)
        res = []
        dirname = None
        for index, frame_data in enumerate(pictures):
            dirname, image_name = os.path.split(frame_data.im_name)
            dirname = dirname[len('images/'):]  # 截取images
            if recreate_videos:
                assert (len(vid_indir) > 0)
                _convert_video_frame_ids(
                    os.path.join(vid_indir, dirname),
                    os.path.join(vid_outdir, dirname))

            res_item = {}

            res_item['image'] = ImageAnnot(folder=os.path.join('images', dirname), name=image_name,
                                           id=index + 1)._asdict()
            res_item['candidates'] = []
            res_item['labeled'] = frame_data.is_labeled
            if res_item['labeled']:
                for box_data in frame_data.boxes:
                    candidate = CandidatesAnnot(track_score=1, track_id=box_data.track_id, det_score=1,
                                                pose_keypoints_2d=box_data.pose.reshape(-1).tolist(),
                                                det_bbox=compute_boxes_from_pose(box_data.pose))._asdict()
                    res_item['candidates'].append(candidate)
            res.append(res_item)
        json_data.append(res)
    with open(out_path, 'w') as fout:
        json.dump(json_data, fout)


def _convert_mat_to_posetrack_json_single_json(annot_dir, out_json_dir, vid_indir, vid_outdir, recreate_videos):
    '''
        分为train和val，两个文件夹，下面每个mat都有与之对应的json
        out_json_dir  xxxx/train    或者 xxxx/val
    '''

    for fpath in tqdm(glob.glob(os.path.join(annot_dir, '*.mat'))):
        json_data = []
        pictures = load_mat(fpath)
        res = []
        # 处理一张图片
        video_name = os.path.basename(fpath).split('.')[0] + '_OpenSVAI.json'
        for index, frame_data in enumerate(pictures):

            dirname, image_name = os.path.split(frame_data.im_name)
            dirname = dirname[len('images/'):]  # 截取images
            res_item = {}

            res_item['image'] = ImageAnnot(folder=os.path.join('images', dirname), name=image_name,
                                           id=index + 1)._asdict()
            res_item['candidates'] = []
            res_item['labeled'] = frame_data.is_labeled
            if res_item['labeled']:
                for box_data in frame_data.boxes:
                    candidate = CandidatesAnnot(track_score=1, track_id=box_data.track_id, det_score=1,
                                                pose_keypoints_2d=box_data.pose.reshape(-1).tolist(),
                                                det_bbox=compute_boxes_from_pose(box_data.pose))._asdict()
                    res_item['candidates'].append(candidate)
            res.append(res_item)
        json_data.append(res)
        out_path = os.path.join(out_json_dir, video_name)
        with open(out_path, 'w') as fout:
            json.dump(json_data, fout)


def convert_to_train_val():
    for split in PoseTrack.splits:
        print('Processing {} split'.format(split))
        _convert_mat_to_posetrack_json(
            PoseTrack.mat_dir.format(split), PoseTrack.out_path.format(split),
            PoseTrack.vid_indir, PoseTrack.vid_outdir, PoseTrack.recreate_videos)


def convert_video_have_json():
    dir = '/media/D/DataSet/PoseTrack2017/{}'
    for split in PoseTrack.splits:
        print('Processing {} split'.format(split))
        _convert_mat_to_posetrack_json_single_json(
            PoseTrack.mat_dir.format(split), dir.format(split),
            PoseTrack.vid_indir, PoseTrack.vid_outdir, PoseTrack.recreate_videos)


if __name__ == '__main__':
    convert_video_have_json()
