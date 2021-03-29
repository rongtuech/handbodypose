import os
import sys
import cv2
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils_common import FileTool
from setting import *

################# utils function ###################
# return shape: (height, width) with the gaussian shape at x, y
def generate_gaussian_heatmap(shape, joint, sigma):
    x, y = joint
    grid_x = np.tile(np.arange(shape[1]), (shape[0], 1)) # matrix shape[0] shape [1]
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() # matrix
    grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
    gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
    return gaussian_heatmap

def generate_heatmaps(img, poses, heatmap_sigma):
    heatmaps = np.zeros((0,) + img.shape[:-1]) # ignore last dim (0, w, h)
    sum_heatmap = np.zeros(img.shape[:-1]) # (w,h) use for background
    for joint_index in range(len(JointType)):
        heatmap = np.zeros(img.shape[:-1])
        for pose in poses:
            if pose[joint_index, 2] > 0: # if visible
                jointmap = generate_gaussian_heatmap(img.shape[:-1], pose[joint_index][:2], heatmap_sigma)
                heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap] # put gaussion point on the heatmap without affect other points
                sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
        heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
    bg_heatmap = 1 - sum_heatmap  # background channel
    heatmaps = np.vstack((heatmaps, bg_heatmap[None]))

    return heatmaps.astype('f')

# return shape: (2, height, width) 2 -> vector of each point
def generate_constant_paf(shape, joint_from, joint_to, paf_width):
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

def generate_pafs(img, poses, paf_sigma):
    pafs = np.zeros((0,) + img.shape[:-1])

    for limb in LIMBS:
        paf = np.zeros((2,) + img.shape[:-1])
        paf_flags = np.zeros(paf.shape) # for constant paf

        for pose in poses:
            joint_from, joint_to = pose[[limb[0].value, limb[1].value]]
            if joint_from[2] > 0 and joint_to[2] > 0: # check visible
                limb_paf = generate_constant_paf(img.shape, joint_from[:2], joint_to[:2], paf_sigma) #[2,368,368]
                limb_paf_flags = limb_paf != 0
                paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)

                paf += limb_paf

        paf[paf_flags > 0] /= paf_flags[paf_flags > 0] # normalize because many paf may stack
        pafs = np.vstack((pafs, paf))
    return pafs.astype('f')

def parse_coco_annotation(img, annotations, transform):
    # input is a list of annotation (each for 1 person)
    # output: poses: np.array( stack of (num join *3) (pos x,y , visible)
    cood = []
    pose_id = []
    join_id = []
    for id, ann in enumerate(annotations):
        ann_pose = np.array(ann['keypoints']).reshape(-1, 3)

        for i, join_index in enumerate(coco_joint_indices):
            if ann_pose[i][2] > 0:
                cood.append((ann_pose[i][0], ann_pose[i][1]))
                pose_id.append(id)
                join_id.append(join_index)

    trans_img, trans_pose_info = transform(image=img, keypoints=(cood, pose_id, join_id))
    poses = np.zeros((len(annotations), len(JointType), 3), dtype=np.int32)
    trans_cood, trans_pose, trans_join = trans_pose_info
    for ind, t_cood in enumerate(trans_cood):
        poses[trans_pose[ind],trans_join[ind].value,0] = t_cood[0]
        poses[trans_pose[ind],trans_join[ind].value,1] = t_cood[1]
        poses[trans_pose[ind],trans_join[ind].value,2] = 2

        # compute neck position
    for id in range(len(annotations)):
        if poses[id,JointType.LeftShoulder.value,2] > 0 and \
                poses[id,JointType.RightShoulder.value,2] > 0:
            poses[id,JointType.Neck.value,0] = int((poses[id,JointType.LeftShoulder.value,0] +
                                              poses[id,JointType.RightShoulder.value,0]) / 2)
            poses[id,JointType.Neck.value,1] = int((poses[id,JointType.LeftShoulder.value,1] +
                                              poses[id,JointType.RightShoulder.value,1]) / 2)
            poses[id,JointType.Neck.value,2] = 2

    return trans_img, poses

################# main dataset #####################

class CocoDataset(Dataset):
    def __init__(self, img_dir, file_info, transform):
        print("read pickle files")
        self.img_dir = img_dir
        self.imgIds = sorted(FileTool.readPickle(file_info["ids"]))
        self.file_info = FileTool.readPickle(file_info["file_info"])
        self.annotations = FileTool.readPickle(file_info["annotations"])
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def get_img_annotation(self, ind):
        img_id = self.imgIds[ind]
        annos = self.annotations[img_id]

        valid_annotations_for_img = []
        for annotation in annos:
            # if too few keypoints or too small
            if annotation['num_keypoints'] >= 5 and annotation['area'] > 32*32:
                valid_annotations_for_img.append(annotation)

        img_path = os.path.join(self.img_dir, self.file_info[img_id]["name"])
        img = cv2.imread(img_path)

        return img, valid_annotations_for_img

    def generate_labels(self, img, poses):
        # input: img, nparry poses [nxnumjointx3]
        # output: transformed image , and its pafs, heatmap
        heatmaps = generate_heatmaps(img, poses, 7)
        pafs = generate_pafs(img, poses, 8)

        return img, pafs, heatmaps

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
            img, annotations = self.get_img_annotation(np.random.randint(len(self.imgIds)))

        trans_img, poses = parse_coco_annotation(img, annotations, self.transform)
        trans_img, pafs, heatmaps = self.generate_labels(trans_img, poses)

        trans_img = self.preprocess(trans_img)
        trans_img = torch.tensor(trans_img)
        pafs = torch.tensor(pafs)
        heatmaps = torch.tensor(heatmaps)

        return trans_img, pafs, heatmaps


class CocoTestDataset(Dataset):
    def __init__(self, img_dir, file_info, transform):
        print("read pickle files")
        self.img_dir = img_dir
        self.imgIds = sorted(FileTool.readPickle(file_info["ids"]))
        self.file_info = FileTool.readPickle(file_info["file_info"])
        self.annotations = FileTool.readPickle(file_info["annotations"])
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def get_img_annotation(self, ind):
        img_id = self.imgIds[ind]
        annos = self.annotations[img_id]

        valid_annotations_for_img = []
        for annotation in annos:
            # if too few keypoints or too small
            if annotation['num_keypoints'] >= 5 and annotation['area'] > 32*32:
                valid_annotations_for_img.append(annotation)

        img_path = os.path.join(self.img_dir, self.file_info[img_id]["name"])
        img = cv2.imread(img_path)

        return img, valid_annotations_for_img

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
            img, annotations = self.get_img_annotation(np.random.randint(len(self.imgIds)))

        trans_img, poses = parse_coco_annotation(img, annotations, self.transform)

        origin_img = trans_img.copy()
        trans_img = self.preprocess(trans_img)
        trans_img = torch.tensor(trans_img)

        return trans_img, origin_img