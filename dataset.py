import os
import sys
import cv2
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from enum import Enum
from utils.utils_common import FileTool

class JointType(Enum):
    Nose = 0
    Neck = 1
    RightShoulder = 2
    RightElbow = 3
    RightHand = 4
    LeftShoulder = 5
    LeftElbow = 6
    LeftHand = 7
    RightWaist = 8
    RightKnee = 9
    RightFoot = 10
    LeftWaist = 11
    LeftKnee = 12
    LeftFoot = 13
    RightEye = 14
    LeftEye = 15
    RightEar = 16
    LeftEar = 17

LIMBS = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
         [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5,9], [6,10], [11,15], [12,16]]


class CocoDataset(Dataset):
    def __init__(self, coco, file_info, image_size):
        self.coco = coco
        print("read pickle files")
        self.imgIds = sorted(FileTool.readPickle(file_info["ids"]))
        self.file_info = FileTool.readPickle(file_info["file_info"])
        self.annotations = FileTool.readPickle(file_info["annotations"])
        self.image_size = image_size

    def __len__(self):
        return len(self.imgIds)

    # return shape: (height, width) with the gaussian shape at x, y
    def generate_gaussian_heatmap(self, shape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1)) # matrix shape[0] shape [1]
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() # matrix
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
        return gaussian_heatmap

    def generate_heatmaps(self, img, poses, heatmap_sigma):
        heatmaps = np.zeros((0,) + img.shape[:-1]) # ignore last dim (0, w, h)
        sum_heatmap = np.zeros(img.shape[:-1]) # (w,h) use for background
        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for pose in poses:
                if pose[joint_index, 2] > 0: # if visible
                    jointmap = self.generate_gaussian_heatmap(img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap] # put gaussion point on the heatmap without affect other points
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))

        return heatmaps.astype('f')

    # return shape: (2, height, width) 2 -> vector of each point
    def generate_constant_paf(self, shape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + shape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance 
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector) # 垂直分量
        grid_x = np.tile(np.arange(shape[1]), (shape[0], 1))
        grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() # grid_x, grid_y用来遍历图上的每一个点
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)

        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width # paf_width : 8

        paf_flag = horizontal_paf_flag & vertical_paf_flag # 合并两个限制条件
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, shape[:-1] + (2,)).transpose(2, 0, 1)

        return constant_paf

    def generate_pafs(self, img, poses, paf_sigma):
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in LIMBS:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for pose in poses:
                joint_from, joint_to = pose[limb]
                if joint_from[2] > 0 and joint_to[2] > 0: # check visible
                    limb_paf = self.generate_constant_paf(img.shape, joint_from[:2], joint_to[:2], paf_sigma) #[2,368,368]
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)

                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0] # normalize because many paf may stack
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def get_img_annotation(self, ind):
        img_id = self.imgIds[ind]
        annos = self.annotations[img_id]

        valid_annotations_for_img = []
        for annotation in annos:
            # if too few keypoints or too small
            if annotation['num_keypoints'] >= 5 and annotation['area'] > 32*32:
                valid_annotations_for_img.append(annotation)

        img_path = os.path.join(self.file_info[img_id]["name"])
        img = cv2.imread(img_path)

        return img, valid_annotations_for_img

    def parse_coco_annotation(self, annotations):
        # input is a list of annotation (each for 1 person)
        # output: poses: np.array( stack of (num join *3) (pos x,y , visible)
        poses = np.zeros((0, len(JointType), 3), dtype=np.int32)

        for ann in annotations:
            ann_pose = np.array(ann['keypoints']).reshape(-1, 3)
            pose = np.zeros((1, len(JointType), 3), dtype=np.int32)

            # convert poses position
            for i, joint_index in enumerate(JointType):
                pose[0][joint_index] = ann_pose[i]

            # compute neck position
            if pose[0][JointType.LeftShoulder][2] > 0 and pose[0][JointType.RightShoulder][2] > 0:
                pose[0][JointType.Neck][0] = int((pose[0][JointType.LeftShoulder][0] +
                                                  pose[0][JointType.RightShoulder][0]) / 2)
                pose[0][JointType.Neck][1] = int((pose[0][JointType.LeftShoulder][1] +
                                                  pose[0][JointType.RightShoulder][1]) / 2)
                pose[0][JointType.Neck][2] = 2

            poses = np.vstack((poses, pose))

        return poses


    def generate_labels(self, img, poses):
        # input: img, nparry poses [nxnumjointx3]
        # output: transformed image , and its pafs, heatmap
        # replace augment code with augmentation code, if use albumentation -> get keypoint ->xy and flatten all keypoints
        # to list, remerge the visible point latter.
        # can define many other class: keypoint type , person belong, visible, remake this
        # TODO: need a function to transform poses array to albumentation ready version and reconvert.
        trans = self.augment_data(image = img, keypoints = poses)
        trans_img, trans_poses = trans["image"], trans["keypoints"]

        heatmaps = self.generate_heatmaps(trans_img, trans_poses, 7)
        pafs = self.generate_pafs(trans_img, trans_poses, 8) # params['paf_sigma']: 8
        return trans_img, pafs, heatmaps

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def __getitem__(self, i):
        img, annotations = self.get_img_annotation(i)

        # if no annotations are available, randomly get another image
        while len(annotations) <= 0:
            i = self.imgIds[np.random.randint(len(self))]
            img, annotations = self.get_img_annotation(i)

        poses = self.parse_coco_annotation(annotations)
        resized_img, pafs, heatmaps = self.generate_labels(img, poses)
        resized_img = self.preprocess(resized_img)
        resized_img = torch.tensor(resized_img)
        pafs = torch.tensor(pafs)
        heatmaps = torch.tensor(heatmaps)

        return resized_img, pafs, heatmaps