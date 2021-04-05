from training_framework_openpose import *
import torch
from utils.utils_torch import parse_args
from torch.utils.data import DataLoader
from loss import compute_loss
from custom_augmentation import COCOTransformation, COCOTransformationTest
from dataset import CocoDataset, CocoTestDataset
from model.mini_model import OpenPoseLightning
from training_framework_openpose import *

if __name__ == "__main__":
    parser = parse_args()
    FIX_HEIGHT = FIX_WIDTH = 368
    # _model
    print("train %s" % parser.model)
    train_file_info = {
        "ids": "../data/coco_bodypose/ids.pkl",
        "file_info": "../data/coco_bodypose/file_infos.pkl",
        "annotations": "../data/coco_bodypose/annotation_ids.pkl"
    }
    val_file_info = {
        "ids": "../data/coco_bodypose/val_ids.pkl",
        "file_info": "../data/coco_bodypose/val_file_infos.pkl",
        "annotations": "../data/coco_bodypose/val_annotation_ids.pkl"
    }

    if parser.state == "train":


        # data augumentation
        data_transforms = COCOTransformation(height=FIX_HEIGHT, width=FIX_WIDTH)
        trainSet = CocoDataset(parser.train, train_file_info, transform=data_transforms)
        # trainSet = CocoDataset(parser.val, val_file_info, transform=data_transforms)
        valSet = CocoDataset(parser.val, val_file_info, transform=data_transforms)
        trainLoader = DataLoader(trainSet, batch_size=20, shuffle=True, num_workers=10)
        valLoader = DataLoader(valSet, batch_size=10, shuffle=False, num_workers=5)

        model = OpenPoseLightning()
        loss = compute_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=parser.lr)

        train_frame = TrainingProcessOpenPose(trainLoader,
                                               valLoader,
                                               optimizer,
                                               loss,
                                               model,
                                               num_epoch=10,
                                               lr=parser.lr,
                                               gpus=parser.gpus,
                                               pretrained_path=parser.weights,
                                               checkpoint_save_path="./_model/model_%s" % (
                                                   parser.model),
                                               is_scheduler=True)
        train_frame.train()

    elif parser.state == "test":
        data_transforms = COCOTransformationTest(height=FIX_HEIGHT, width=FIX_WIDTH)
        testSet = CocoTestDataset(parser.test, val_file_info, transform=data_transforms)
        testLoader = DataLoader(testSet, batch_size=1, shuffle=True)
        model = OpenPoseLightning()
        eval_framework = EvaluationOpenPose(testLoader,
                                            model,
                                            parser.weights)

        eval_framework.test(num_show=6)