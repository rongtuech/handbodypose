import os
import cv2
import numpy as np
import math
import torch


def normalize_in(img: np.array) -> np.array:
    """"""
    img = img.astype(np.float32)
    img /= 256.0
    img -= 0.5
    return img


def normalize_gt(img: np.array) -> np.array:
    """"""
    img = img.astype(np.float32)
    img /= 255.0
    return img


def add_border(img: np.array, size_x: int = 128, size_y: int = 128) -> (np.array, int, int):
    """Add border to image, so it will divide window sizes: size_x and size_y"""
    max_y, max_x = img.shape[:2]
    border_y = 0
    if max_y % size_y != 0:
        border_y = (size_y - (max_y % size_y) + 1) // 2
        img = cv2.copyMakeBorder(img, border_y, border_y, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    border_x = 0
    if max_x % size_x != 0:
        border_x = (size_x - (max_x % size_x) + 1) // 2
        img = cv2.copyMakeBorder(img, 0, 0, border_x, border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img, border_y, border_x

def padding_to_right_shape(img: np.array, size_x: int = 128, size_y: int = 128) -> np.array:
    height, width, _ = img.shape
    pad_h = math.ceil(height/size_y)*size_y - height
    pad_w = math.ceil(width/size_x)*size_x - width

    return cv2.copyMakeBorder(img,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT, value=[0,0,0])

def split_img(img: np.array, size_x: int = 128, size_y: int = 128) -> [np.array]:
    """Split image to parts (little images).
    Walk through the whole image by the window of size size_x * size_y without overlays and
    save all parts in list. Images sizes should divide window sizes.
    """
    max_y, max_x = img.shape[:2]
    parts = []
    curr_y = 0
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            parts.append(img[curr_y:curr_y + size_y, curr_x:curr_x + size_x])
            curr_x += size_x
        curr_y += size_y

    return parts


def combine_imgs(imgs: [np.array], max_y: int, max_x: int) -> np.array:
    """Combine image parts to one big image.
    Walk through list of images and create from them one big image with sizes max_x * max_y.
    If border_x and border_y are non-zero, they will be removed from created image.
    The list of images should contain data in the following order:
    from left to right, from top to bottom.
    """
    img = np.zeros((max_y, max_x), np.float)
    size_y, size_x = imgs[0].shape
    curr_y = 0
    i = 0
    # TODO: rewrite with generators.
    while (curr_y + size_y) <= max_y:
        curr_x = 0
        while (curr_x + size_x) <= max_x:
            try:
                img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] = imgs[i]
            except:
                i -= 1
            i += 1
            curr_x += size_x
        curr_y += size_y
    return img


def preprocess_img(img: np.array) -> np.array:
    """Apply bilateral filter to image."""
    #img = cv2.bilateralFilter(img, 5, 50, 50) TODO: change parameters.
    return img


def process_unet_img(img: np.array, model, batchsize: int = 20) -> np.array:
    """Split image to 128x128 parts and run U-net for every part."""
    img, border_y, border_x = add_border(img)
    img = normalize_in(img)
    parts = split_img(img)
    parts = np.array(parts)
    parts.shape = (parts.shape[0], parts.shape[1], parts.shape[2], 1)
    parts = model.predict(parts, batchsize)
    tmp = []
    for part in parts:
        part.shape = (128, 128)
        tmp.append(part)
    parts = tmp
    img = combine_imgs(parts, img.shape[0], img.shape[1])
    img = img[border_y:img.shape[0] - border_y, border_x:img.shape[1] - border_x]
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def postprocess_img(img: np.array) -> np.array:
    """Apply Otsu threshold to image."""
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def binarize_img(img: np.array, model, batchsize: int = 20) -> np.array:
    """Binarize image, using U-net, Otsu, bottom-hat transform etc."""
    img = preprocess_img(img)
    img = process_unet_img(img, model, batchsize)
    img = postprocess_img(img)
    return img


def mkdir_s(path: str):
    """Create directory in specified path, if not exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def rescale_image(image: np.array, h:int = None, w:int = None) -> np.array:
    c_h, c_w = image.shape[:2]
    if w is None and h is None:
        return image
    elif w is None:
        h_ratio = w_ratio = float(h) / c_h
    elif h is None:
        h_ratio = w_ratio = float(w) / c_w
    else:
        h_ratio = float(h) / c_h
        w_ratio = float(w) / c_w

    return cv2.resize(image, (int(c_w * w_ratio), int(c_h * h_ratio)), interpolation=cv2.INTER_CUBIC)


def padding_image(image: np.array, h:int = None, w:int = None, color_value=None) -> np.array:
    c_h, c_w = image.shape[:2]

    pad_h = math.ceil(c_h / h) * h - c_h
    pad_w = math.ceil(c_w / w) * w - c_w

    return cv2.copyMakeBorder(image, 0,pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value= color_value)

def gray_scale2rbg(img: np.array) -> np.array:
    img = np.clip(img, a_min=0, a_max=255)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def tensor2npImage(img_tensor: np.array) -> np.array:
    img_tensor = np.swapaxes(img_tensor, 0,1)
    img_tensor = np.swapaxes(img_tensor, 1,2)
    img_tensor = img_tensor * 255
    img_tensor = np.clip(img_tensor, a_min=0, a_max=255)
    return img_tensor

body_edges = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle


def draw_poses_for_coco(img, poses_2d):
    # get only the biggest pose of image
    max_size =0
    max_index = 0
    size_hand = 0
    for pose_id in range(len(poses_2d)):
        pose = np.array(poses_2d[pose_id][0:-1]).reshape((-1, 3)).transpose()
        max_x, min_x = np.max(pose[1, :]), np.min(pose[1, :])
        max_y, min_y = np.max(pose[0, :]), np.min(pose[0, :])
        temp_size = abs((max_x-min_x)*(max_y-min_y))
        if  temp_size > max_size:
            max_size=temp_size
            max_index = pose_id
            size_hand = max((max_x-min_x),(max_y-min_y))
            size_hand = size_hand //8

    poses_2d = [poses_2d[max_index]]
    for pose_id in range(len(poses_2d)):
        pose = np.array(poses_2d[pose_id][0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2, :] > 0
        for edge in body_edges:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(img, tuple(pose[0:2, edge[0]].astype(int)), tuple(pose[0:2, edge[1]].astype(int)),
                         (255, 255, 0), 2, cv2.LINE_AA)

        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                cv2.circle(img, tuple(pose[0:2, kpt_id].astype(int)), 2, (0, 255, 255), -1, cv2.LINE_AA)

        # get hand position
        if was_found[4] and was_found[5]:
            l_h = pose[0:2,5]+ (pose[0:2,5] - pose[0:2,4])/3
            cv2.rectangle(img,
                          (int(l_h[0] - size_hand / 2), int(l_h[1] - size_hand / 2)),
                          (int(l_h[0] + size_hand / 2), int(l_h[1] + size_hand / 2)),
                          (0,0, 255),1)

        if was_found[10] and was_found[11]:
            r_h = pose[0:2,11] + (pose[0:2,11] - pose[0:2,10])/3
            cv2.rectangle(img,
                          (int(r_h[0] - size_hand/2),int(r_h[1] - size_hand/2)),
                           (int(r_h[0] + size_hand/2),int(r_h[1] + size_hand/2)),
                          (0, 0, 255),1)
hand_edges = [[0, 1],
     [1, 2], [2, 3], [3, 4], # nose - l_eye - l_ear
     [0, 5], [5, 6],[6, 7],[7, 8],  # nose - r_eye - r_ear
     [0, 9], [9,10], [10, 11],[11, 12],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 13], [13, 14], [14, 15],[15, 16],  # neck - r_shoulder - r_elbow - r_wrist
     [0, 17], [17, 18], [18, 19],[19, 20]]  # neck - r_hip - r_knee - r_ankle


def draw_hand_pose(img, hand_points):
    for edge in hand_edges:
        if hand_points[edge[0]] is not None and hand_points[edge[1]]:
            cv2.line(img, (int(hand_points[edge[0]][0]), int(hand_points[edge[0]][1])),
                     int(hand_points[edge[1]][0]), int(hand_points[edge[1]][1]),
                     (255, 255, 0), 2, cv2.LINE_AA)

    for point in range(len(hand_points)):
        if point is not None:
            cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1, cv2.LINE_AA)
