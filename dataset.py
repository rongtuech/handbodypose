import os
import sys
import cv2
import math
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.utils_common import FileTool
from setting import *
import json

################# utils function ###################
# return shape: (height, width) with the gaussian shape at x, y
def generate_gaussian_heatmap(shape, joint, sigma):
    x, y = joint
    grid_x = np.tile(np.arange(shape[1]), (shape[0], 1)) # matrix shape[0] shape [1]
    grid_y = np.tile(np.arange(shape[0]), (shape[1], 1)).transpose() # matrix
    grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
    gaussian_heatmap = np.exp(-0.5 * grid_distance / sigma**2)
    return gaussian_heatmap

def generate_heatmaps(img, poses, heatmap_sigma, join_type):
    heatmaps = np.zeros((0,) + img.shape[:-1]) # ignore last dim (0, w, h)
    sum_heatmap = np.zeros(img.shape[:-1]) # (w,h) use for background
    for joint_index in range(len(join_type)):
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

def generate_pafs(img, poses, paf_sigma, join_connect):
    pafs = np.zeros((0,) + img.shape[:-1])

    for limb in join_connect:
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

def parse_hand_annotation(img, annotations, transform):
    cood = []
    join_id = []
    for i, join_index in enumerate(hand_join_indices):
        if annotations[i,2] > 0:
            cood.append((annotations[i,0], annotations[i,1]))
            join_id.append(join_index)

    trans_img, trans_pose_info = transform(image=img, keypoints=(cood, join_id))
    poses = np.zeros((1, len(HandJointType), 3), dtype=np.int32)
    trans_cood, trans_join = trans_pose_info
    for ind, t_cood in enumerate(trans_cood):
        poses[0, trans_join[ind].value, 0] = t_cood[0]
        poses[0, trans_join[ind].value, 1] = t_cood[1]
        poses[0, trans_join[ind].value, 2] = 2

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
        heatmaps = generate_heatmaps(img, poses, 7, JointType)
        pafs = generate_pafs(img, poses, 8, LIMBS)

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


################### hand dataset #################

class HandDataset(Dataset):
    def __init__(self, img_dir, transform, is_visualize=False):
        print("read pickle files")
        self.img_dir = img_dir
        self.imgIds = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        print(len(self.imgIds))
        self.transform = transform
        self.is_visualize = is_visualize

    def __len__(self):
        return len(self.imgIds)

    def get_img_annotation(self, ind):
        img_path = self.imgIds[ind]
        image = cv2.imread(img_path)
        annos = json.loads(FileTool.read_text_file(img_path.replace(".jpg",".json"))[0])

        handpoints = np.array(annos["hand_pts"])
        hand_center = np.array(annos["hand_box_center"])
        max_x, min_x = np.max(handpoints[:,0]), np.min(handpoints[:,0])
        max_y, min_y = np.max(handpoints[:,1]), np.min(handpoints[:,1])
        size = int(max_x - min_x)*3 if (max_x - min_x) > (max_y - min_y) \
            else int(max_y - min_y)*3

        image = cv2.copyMakeBorder( image, size//2, size//2, size//2, size//2,
                                    cv2.BORDER_REPLICATE)
        handpoints[:,0:2] = handpoints[:,0:2] + size//2
        hand_center = hand_center + size//2

        image = image[int(hand_center[1] - size/2):int(hand_center[1] + size/2),
                int(hand_center[0] - size/2):int(hand_center[0] + size/2),:]

        handpoints[:,0] = handpoints[:,0] - int(hand_center[0] - size/2)
        handpoints[:,1] = handpoints[:,1] - int(hand_center[1] - size/2)

        return image, handpoints

    def generate_labels(self, img, poses):
        # input: img, nparry poses [nxnumjointx3]
        # output: transformed image , and its pafs, heatmap
        heatmaps = generate_heatmaps(img, poses, 7, HandJointType)
        pafs = generate_pafs(img, poses, 8, HANDLINES)

        return img, pafs, heatmaps

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def __getitem__(self, i):
        img, annotations = self.get_img_annotation(i)

        trans_img, poses = parse_hand_annotation(img, annotations, self.transform)
        trans_img, pafs, heatmaps = self.generate_labels(trans_img, poses)

        trans_img = self.preprocess(trans_img)
        trans_img = torch.tensor(trans_img)
        pafs = torch.tensor(pafs)
        heatmaps = torch.tensor(heatmaps)

        return trans_img, pafs, heatmaps


class HandTestset(Dataset):
    def __init__(self, img_dir, transform):
        print("read pickle files")
        self.img_dir = img_dir
        self.imgIds = glob.glob(os.path.join(self.img_dir,"*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.imgIds)

    def get_img_annotation(self, ind):
        img_path = self.imgIds[ind]
        image = cv2.imread(img_path)
        annos = json.loads(FileTool.readPickle(img_path.replace(".jpg",".json")))

        handpoints = np.array(annos["hand_pts"])
        hand_center = annos["hand_box_center"]
        max_x, min_x = np.max(handpoints[:,0]), np.min(handpoints[:,0])
        max_y, min_y = np.max(handpoints[:,1]), np.min(handpoints[:,1])
        size = int(max_x - min_x)*3 if (max_x - min_x) > (max_y - min_y) \
            else int(max_y - min_y)*3

        image = image[int(hand_center[0] - size/2):int(hand_center[0] + size/2),
                int(hand_center[1] - size/2):int(hand_center[1] + size/2),:]
        handpoints[:,0] = handpoints[:,0] - int(hand_center[0] - size/2)
        handpoints[:,1] = handpoints[:,1] - int(hand_center[1] - size/2)


        return image, handpoints

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)
        return x_data

    def __getitem__(self, i):
        img, annotations = self.get_img_annotation(i)

        trans_img, poses = parse_hand_annotation(img, annotations, self.transform)

        origin_img = trans_img.copy()
        trans_img = self.preprocess(trans_img)
        trans_img = torch.tensor(trans_img)

        return trans_img,  origin_img