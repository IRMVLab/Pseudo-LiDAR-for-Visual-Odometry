# -*- coding:UTF-8 -*-

import os
import yaml
import torch
import argparse
import numpy as np
import torch.utils.data as data
import os.path as osp
import lib.utils.calibration as calibration

from PIL import Image
from tools.euler_tools import euler2quat, mat2euler
from tools.points_process import aug_matrix, limited_points, filter_points

# author:Zhiheng Feng
# contact: fzhsjtu@foxmail.com
# datetime:2021/10/21 20:04
# software: PyCharm

"""
文件说明：数据集读取

"""


class points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 150000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6], config: argparse.Namespace = None):
        """

        :param train: 0训练集，1验证集，2测试集
        :param data_dir_list: 数据集序列
        :param config: 配置参数
        """

        data_dir_list.sort()
        self.num_point = num_point
        self.is_training = is_training
        self.args = config
        self.data_list = data_dir_list
        self.lidar_root = config.data_root
        self.root_image = osp.join('/dataset', 'data_odometry_color')
        self.data_len_sequence = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            fn3 = index_
            fn4 = index_
            c1 = index_
            c2 = index_
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            fn3 = index_ - 1
            fn4 = index_
            c1 = index_ - 1
            c2 = index_

        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)### 
        
        pose_path = 'pose/' + sequence_str_list[index_index] + '_diff.npy'
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index])
        calib = os.path.join(self.root_image, sequence_str_list[index_index],'calib')
        datapath_left = os.path.join(self.root_image, sequence_str_list[index_index], 'image_2')
        pose = np.load(pose_path)

        fn1_dir = os.path.join(lidar_path, '{:06d}.npy'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.npy'.format(fn2))
        fn3_dir = os.path.join(datapath_left, '{:06d}.png'.format(fn3))
        fn4_dir = os.path.join(datapath_left, '{:06d}.png'.format(fn4))
        c1_dir = os.path.join(calib, '{:06d}.txt'.format(c1))
        c2_dir = os.path.join(calib, '{:06d}.txt'.format(c2))
        
        point1 = np.load(fn1_dir)
        point2 = np.load(fn2_dir)

        img3 = Image.open(fn3_dir).convert('RGB')
        img4 = Image.open(fn4_dir).convert('RGB')
        width3, height3 = img3.size
        width4, height4 = img4.size

        img3 = np.array(img3).astype(np.float)
        img3 = img3 / 255.0
        img3 -= self.mean  # 图像归一化
        img3 /= self.std
        imback3 = np.zeros([384, 1248, 3], dtype=np.float)
        imback3[:img3.shape[0], :img3.shape[1], :] = img3
        imback3 = imback3.transpose(2, 0, 1)

        img4 = np.array(img4).astype(np.float)
        img4 = img4 / 255.0
        img4 -= self.mean
        img4 /= self.std
        imback4 = np.zeros([384, 1248, 3], dtype=np.float)
        imback4[:img4.shape[0], :img4.shape[1], :] = img4
        imback4 = imback4.transpose(2, 0, 1)

        calib1 = calibration.Calibration(c1_dir)
        calib2 = calibration.Calibration(c2_dir)

        xy = np.load('./ground_truth_pose/xy.npy')

        pos1 = np.zeros((self.num_point, 3))
        pos2 = np.zeros((self.num_point, 3))

        pos1[ :point1.shape[0], :] = point1[:, :3]
        pos2[ :point2.shape[0], :] = point2[:, :3]
        
        T_diff = pose[index_:index_ + 1, :]  ##### read the transformation matrix

        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)  ##1*4
        T_diff = np.concatenate([T_diff, filler], axis=0)  # 4*4

        T_gt = np.matmul(Tr_inv, T_diff)
        T_gt = np.matmul(T_gt, Tr)

        pos1 = pos1.astype(np.float32)
        pos2 = pos2.astype(np.float32)
        
        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)
        
        batch1 = {'pose':pos1,'calib':calib1,'image':imback3,'xy':xy}
        batch2 = {'pose':pos2,'calib':calib2,'image':imback4,'xy':xy}
        return batch1, batch2, T_gt, T_trans, T_trans_inv, Tr
        
    def collate_batch(self, batch):
        batch_size = batch.__len__()
        data = batch[0]
        depth_dict1 = {}
        depth_dict2 = {}
        depth_batch = {}

        T_gt = np.concatenate([batch[k][2][np.newaxis, ...] for k in range(batch_size)], axis=0)
        T_trans = np.concatenate([batch[k][3][np.newaxis, ...] for k in range(batch_size)], axis=0)
        T_trans_inv = np.concatenate([batch[k][4][np.newaxis, ...] for k in range(batch_size)], axis=0)
        Tr = np.concatenate([batch[k][5][np.newaxis, ...] for k in range(batch_size)], axis=0)
        
    
        for key in data[0].keys():

            if isinstance(batch[0][0][key], np.ndarray):
                depth_dict1[key] = np.concatenate([batch[k][0][key][np.newaxis, ...] for k in range(batch_size)], axis=0)
                depth_dict2[key] = np.concatenate([batch[k][1][key][np.newaxis, ...] for k in range(batch_size)], axis=0)
                depth_batch[key] = np.concatenate([depth_dict1[key],depth_dict2[key]], axis=0)
            else:
                depth_dict1[key] = [batch[k][0][key] for k in range(batch_size)]
                depth_dict2[key] = [batch[k][1][key] for k in range(batch_size)]
                depth_batch[key] = depth_dict1[key] + depth_dict2[key]


        return depth_batch, T_gt, T_trans, T_trans_inv, Tr
    
    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num
