import numpy as np
import cv2
from parse_poses import parse_poses
import torch


class Pose:
    """
    store and parsing for drawing, optical flow
    """
    def __init__(self, image_scale,  base_on_prob = False):
        self.image_scale = image_scale
        self.poses_list = []
        self.poses_prob = []
        self.hand_window = []
        self.head_window = []

        # get best only
        self.base_on_prob = base_on_prob

    def parser_pose(self, paf, heatmap):
        # get pose list ([keypoints x 3], and prob)
        self.poses_list, self.poses_prob = parse_poses((paf, heatmap), self.image_scale)
        self.hand_window = []
        self.head_window = []
        # get the best pose to index 0
        if len(self.poses_list) > 0:
            self.get_best_pose()

            # get the hand and head window 0
            self.get_hand_head_window()

    def get_best_pose(self):
        # get only the biggest pose of image
        if self.base_on_prob:
            self.poses_list = [x for _, x in sorted(zip(self.poses_prob, self.poses_list))]
        else:
            max_size = 0
            max_index = 0
            for pose_id in range(len(self.poses_list)):
                max_x, min_x = np.max(self.poses_list[pose_id][:, 1]), np.min(self.poses_list[pose_id][:, 1])
                max_y, min_y = np.max(self.poses_list[pose_id][:, 0]), np.min(self.poses_list[pose_id][:, 0])
                temp_size = abs((max_x - min_x) * (max_y - min_y))
                if temp_size > max_size:
                    max_size = temp_size
                    max_index = pose_id
            self.poses_list[0], self.poses_list[max_index] = self.poses_list[max_index], self.poses_list[0]

    def get_hand_head_window(self):
        for pose_id in range(len(self.poses_list)):
            current_pose = self.poses_list[pose_id]
            max_x, min_x = np.max(current_pose[:, 1]), np.min(current_pose[:, 1])
            max_y, min_y = np.max(current_pose[:, 0]), np.min(current_pose[:, 0])
            size_hand = max((max_x - min_x), (max_y - min_y))
            size_hand = size_hand // 8

            was_found = current_pose[:, 2] > 0

            handtuple = []
            # get hand position
            if was_found[4] and was_found[5]:
                l_h = current_pose[5, 0:2] + (current_pose[5, 0:2] - current_pose[4, 0:2]) / 3
                handtuple.append(((
                                      int(l_h[0] - size_hand / 2),
                                      int(l_h[1] - size_hand / 2)
                                  ),
                                  (
                                      int(l_h[0] + size_hand / 2),
                                      int(l_h[1] + size_hand / 2)
                                  )
                                ))

            if was_found[10] and was_found[11]:
                r_h = current_pose[11, 0:2] + (current_pose[11, 0:2] - current_pose[10, 0:2]) / 3
                handtuple.append(((
                                      int(r_h[0] - size_hand / 2),
                                      int(r_h[1] - size_hand / 2)
                                  ),
                                  (
                                      int(r_h[0] + size_hand / 2),
                                      int(r_h[1] + size_hand / 2)
                                  )))
            self.hand_window.append(handtuple)

            # if was_found[1]:
            #     center_points = current_pose[1,0:2]
            #     self.head_window.append(((
            #                               int(center_points[0] - size_hand / 2),
            #                               int(center_points[1] - size_hand / 2)
            #                           ),
            #                           (
            #                               int(center_points[0] + size_hand / 2),
            #                               int(center_points[1] + size_hand / 2)
            #                           )))
            # else:
            #     self.head_window.append([])

    def get_hand_head_images(self, origin_image,ratio, pad):
        hand_img = []
        w,h = 0,0
        if len(self.hand_window) >0:
            for window in self.hand_window[0]:
                temp_img = origin_image[int(window[0][1]*ratio):int(window[1][1]*ratio),
                                int(max(window[0][0]-pad,0)*ratio): int(max(window[1][0]-pad,0)*ratio)]

                # print(int(window[0][1]*ratio),int(window[1][1]*ratio),
                #             int((window[0][0]-pad)*ratio), int((window[1][0]-pad)*ratio))
                if len(hand_img) ==0:
                    w,h, _ = temp_img.shape
                else:
                    w_t,h_t, _ = temp_img.shape
                    if w_t != w:
                        temp_img = temp_img[:w,:h]
                hand_img.append(temp_img)
        if len(hand_img) == 0:
            return None
        else:
            img_sum = cv2.hconcat(hand_img)
            return img_sum
