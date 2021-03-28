from training_framework import TrainingProcess, EvaluationProcess
from tqdm import tqdm
from utils.utils_torch import transform_torch_vars, transform_torch_targets
import torch
from utils.utils_torch import parse_args
from torch.utils.data import DataLoader
from loss import compute_loss
from custom_augmentation import COCOTransformation, COCOTransformationTest
from dataset import CocoDataset
from model.mini_model import PoseEstimationWithMobileNet
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class TrainingProcessBodyPose(TrainingProcess):
    def __init__(self, *args,
                 is_triplet=False,
                 **kwargs):
        """
         config data loader and gpus using for training.
        :param training_dataloader:
        :param validate_dataloaer:
        :param optimizer:
        :param loss_func:
        :param gpus:
        # :param _model:
        # """
        super().__init__(*args, **kwargs)
        self.is_triplet = is_triplet
        self.global_step_lr = 0
        self.lr_this_step =0
        self.last_model_path = None

    def _show_log(self, train_info, pbar):
        self.writer.add_scalars("loss", {
            "Loss/train_sum": train_info[0],
            "Loss/train_paf": train_info[1],
            "Loss/train_heatmap": train_info[2],
            "Loss/lr": self.lr_this_step
        })
        pbar.set_postfix(loss='%.4f' %train_info[0],
                         loss_paf = '%.4f' %train_info[1],
                         loss_heatmap = '%.4f' %train_info[2],
                         lr = "%.6f" % self.lr_this_step)


    def train(self):
        """
        1) set parallel mode
        2) load pretrained data
        3) set train mode for _model and optimizer and loss
        4) loop with num epoch
        :return:
        """
        # init data
        smooth_loss_sum = None
        smooth_loss_paf = None
        smooth_loss_heatmap = None
        self._show_init_data()
        # setting
        self.model.train()
        ###### training ######
        for epoch_index in range(self.num_epoch):
            pbar = tqdm(enumerate(self.training_data),total= len(self.training_data), ncols=-1)
            for batch_id, data in pbar:
                loss_sum, loss_paf, loss_heatmap = self._train_per_batch(data)

                smooth_loss_sum = 0.5*smooth_loss_sum + 0.5* loss_sum \
                    if smooth_loss_sum is not None else loss_sum
                smooth_loss_paf = 0.5*smooth_loss_paf + 0.5* loss_paf \
                    if smooth_loss_paf is not None else loss_paf
                smooth_loss_heatmap = 0.5*smooth_loss_heatmap + 0.5* loss_heatmap \
                    if smooth_loss_heatmap is not None else loss_heatmap
                # log data
                self._show_log([smooth_loss_sum, smooth_loss_paf, smooth_loss_heatmap],pbar)

            # validate and log
            if epoch_index % 1 == 0:
                # run validation part
                val_loss = self._validate()
                self._save_model(val_loss, epoch_index)

    def _train_per_batch(self, data):
        """
            private func for batch train
        :param data:
        :param label:
        :return:
        """
        resized_img, paf_t, heatmap_t = data
        # transform data #
        resized_img = transform_torch_vars(resized_img, self.is_cuda)
        paf_t = transform_torch_targets(paf_t, self.is_cuda)
        heatmap_t = transform_torch_targets(heatmap_t, self.is_cuda)
        self.optimizer.zero_grad()

        paf, heatmap = self.model(resized_img)
        loss_sum, loss_paf, loss_heatmap = self.loss_func(paf, heatmap, paf_t, heatmap_t)

        if self.is_apex:
            # with apex.amp.scale_loss(loss_sum, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            pass
        else:
            loss_sum.backward()
        self.optimizer.step()

        #update lr
        if self.scheduler is not None:
            self.lr_this_step = self.lr * self.scheduler.get_lr(self.global_step_lr, 0.03)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr_this_step
            self.global_step_lr +=1

        return loss_sum.item(), loss_paf, loss_heatmap

    def _validate(self):
        """
        private validate func
        :return: average of validated loss
        """
        self.model.eval()
        loss_val = 0
        with torch.no_grad():
            for data in tqdm(self.validate_data):
                resized_img, paf_t, heatmap_t = data
                resized_img = transform_torch_vars(resized_img, self.is_cuda)
                paf_t = transform_torch_targets(paf_t, self.is_cuda)
                heatmap_t = transform_torch_targets(heatmap_t, self.is_cuda)

                paf, heatmap = self.model(resized_img)

                loss_sum, _, _ = self.loss_func(paf, heatmap, paf_t, heatmap_t)
                loss_val += loss_sum.item()

        self.model.train()

        return loss_val / len(self.validate_data)

    def _save_model(self, val_loss, epoch_index):
        print("epoch=%d; val_loss=%.4f"%(epoch_index, val_loss))
        # early stop.
        if self.best_current_loss > val_loss:
            self.best_current_loss = val_loss
            print("save the best _model \t new best loss \t %.5f to file %s" % (self.best_current_loss,self.checkpoint_path))
            current_path = self.checkpoint_path + "epoch_%d_loss_%.4f.pth"%(epoch_index,val_loss)
            if self.is_cuda:
                torch.save(self.model.state_dict(),
                           self.checkpoint_path + "epoch_%d_loss_%.4f.pth"%(epoch_index,val_loss))
            else:
                torch.save(self.model.state_dict(),
                           self.checkpoint_path + "epoch_%d_loss_%.4f.pth"%(epoch_index,val_loss))
            if self.last_model_path is not None:
                os.remove(self.last_model_path)
            self.last_model_path = current_path


import cv2
from parse_poses import *
from utils.utils_image import draw_poses_for_coco

class EvaluationCOCOPose(EvaluationProcess):
    def test(self, num_show = 3):
        self.model.eval()
        current_batch = 0
        with torch.no_grad():
            for data in tqdm(self.test_data):
                resized_img, paf_t, heatmap_t = data
                resized_img = transform_torch_vars(resized_img, self.is_cuda)

                paf, heatmap = self.model(resized_img)
                # view paf and heatmap here

                # parse paf and heatmap here
                self.parser_output((paf, heatmap), resized_img, current_batch)

                current_batch+=1
                if current_batch >  num_show:
                    return

    def view_paf_heatmap(self, paf_batch, heatmap_batch,ind):
        paf_batch = paf_batch.detach().cpu().numpy()
        paf_batch = paf_batch[0]*255
        heatmap_batch = heatmap_batch.detach().cpu().numpy()
        heatmap_batch = heatmap_batch[0]*255
        for i,heatmap in enumerate(heatmap_batch):
            cv2.imwrite("heatmap_%d_%d.jpg" % (ind, i), heatmap)

        for i, paf in enumerate(paf_batch):
            cv2.imwrite("paf_%d_%d.jpg"%(ind, i), paf)

    def parser_output(self, pred, img, ind):
        current_time = cv2.getTickCount()
        poses_2d = parse_poses(pred, (1.0,1.0))

        draw_poses_for_coco(img, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        cv2.putText(img, 'FPS: {}'.format(current_time),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imwrite("img_with_pose_%d.jpg"%ind, img)

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

        model = PoseEstimationWithMobileNet()
        loss = compute_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=parser.lr)

        train_frame = TrainingProcessBodyPose(trainLoader,
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
        testSet = CocoDataset(parser.test, val_file_info, transform=data_transforms)
        testLoader = DataLoader(testSet, batch_size=1, shuffle=False)
        model = PoseEstimationWithMobileNet()
        eval_framework = EvaluationCOCOPose(testLoader,
                                            model,
                                            parser.weights,
                                            gpus = parser.gpus)

        eval_framework.test()