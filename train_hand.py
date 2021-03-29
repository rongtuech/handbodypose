from training_framework_openpose import *
import torch
from utils.utils_torch import parse_args
from torch.utils.data import DataLoader
from loss import compute_loss
from custom_augmentation import HandTransformation
from dataset import HandDataset
from model.mini_model import PoseEstimationWithMobileNet
import setting

if __name__ == "__main__":
    parser = parse_args()
    FIX_HEIGHT = FIX_WIDTH = 256
    # _model
    print("train %s" % parser.model)

    if parser.state == "train":
        # data augumentation
        data_transforms = HandTransformation(height=FIX_HEIGHT, width=FIX_WIDTH)
        trainSet = HandDataset(parser.train, transform=data_transforms)
        valSet = HandDataset(parser.val, transform=data_transforms)
        trainLoader = DataLoader(trainSet, batch_size= 10, shuffle=False, num_workers=3)
        valLoader = DataLoader(valSet, batch_size=8, shuffle=False, num_workers=2)

        model = PoseEstimationWithMobileNet(num_pafs=2*len(setting.HANDLINES),
                                            num_heatmaps=len(setting.HandJointType)+1)
        loss = compute_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=parser.lr)

        train_frame = TrainingProcessOpenPose(trainLoader,
                                               valLoader,
                                               optimizer,
                                               loss,
                                               model,
                                               num_epoch=50,
                                               lr=parser.lr,
                                               gpus=parser.gpus,
                                               pretrained_path=parser.weights,
                                               checkpoint_save_path="./_model/model_%s" % (
                                                   parser.model),
                                               is_scheduler=True,
                                              is_apex=False)
        train_frame.train()