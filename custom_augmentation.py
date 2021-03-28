import cv2
from albumentations import Compose, PadIfNeeded, ShiftScaleRotate, ImageCompression, KeypointParams, \
    LongestMaxSize
from albumentations.imgaug.transforms import IAAAffine


class COCOTransformation:
    def __init__(self, width, height):
        self.aug = Compose([
            ShiftScaleRotate(p=0.5, rotate_limit=5, scale_limit=0.05, border_mode=cv2.BORDER_CONSTANT),
            ImageCompression(quality_lower=95, quality_upper=100, p=1),
            IAAAffine(shear=0.2, always_apply=False, p=0.3),
            LongestMaxSize(max_size=width if width > height else height),
            PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT)
        ], keypoint_params=KeypointParams(format='xy', label_fields=['pose_id', "join_id"],
                                          remove_invisible=True))

    def __call__(self, image, keypoints):
        cood, pose, join = keypoints
        transformed = self.aug(image=image, keypoints=cood, pose_id=pose, join_id=join)
        return transformed["image"], (transformed['keypoints'], transformed['pose_id'], \
               transformed['join_id'])


class COCOTransformationTest:
    def __init__(self, width, height):
        self.aug = Compose([
            LongestMaxSize(max_size=width if width > height else height),
            PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT)
        ], keypoint_params=KeypointParams(format='xy', label_fields=['pose_id', "join_id"],
                                          remove_invisible=True))

    def __call__(self, image, keypoints):
        cood, pose, join = keypoints
        transformed = self.aug(image=image, keypoints=cood, pose_id=pose, join_id=join)
        return transformed["image"], (transformed['keypoints'], transformed['pose_id'], \
               transformed['join_id'])