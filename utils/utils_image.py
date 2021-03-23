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
