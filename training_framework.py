import os
import torch
from tqdm import tqdm
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utils.utils_torch import *
from utils.utils_common import FileTool
from utils import utils_image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
# import apex
from schedules import WarmupLinearSchedule

class TrainingProcess:
    def __init__(self,
                 training_dataloader,
                 validate_dataloaer,
                 optimizer,
                 loss_func,
                 model,
                 num_epoch = 100,
                 lr = 0.0002,
                 gpus=None,
                 pretrained_path = None,
                 checkpoint_save_path = "best_model.pt",
                 is_apex = True,
                 is_scheduler = True):
        """
         config data loader and gpus using for training.
        :param training_dataloader:
        :param validate_dataloaer:
        :param optimizer:
        :param loss_func:
        :param gpus:
        # :param _model:
        # """
        # init data
        self.training_data = training_dataloader
        self.validate_data = validate_dataloaer
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.model = model
        self.checkpoint_path = checkpoint_save_path

        # support vars
        self.name_model = model.__class__.__name__
        self.writer = SummaryWriter()
        self.best_current_loss = 1000000000
        self.current_delay_overfit = 0
        self.is_apex = is_apex
        self.lr = lr
        self.num_epoch = num_epoch
        self.scheduler = WarmupLinearSchedule(warmup=0.03, t_total=len(training_dataloader)* num_epoch) if is_scheduler else None
        # load pre-trained _model
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        if pretrained_path is not None:
            print("loaded model %s"%pretrained_path)
            self.model.load_state_dict(torch.load(pretrained_path))

        # setting cuda if needed
        self.gpus, self.model = setting_cuda(gpus, self.model)

        if self.is_apex:
            # self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level="O1",
            #                                                  verbosity=0)
            pass
        self.is_cuda = len(self.gpus) >= 1

    def train(self, num_epoch, num_validate_per_batch = 10, num_show_log = 10):
        """
        1) set parallel mode
        2) load pretrained data
        3) set train mode for _model and optimizer and loss
        4) loop with num epoch
        :param num_epoch:
        :param num_validate_per_batch:
        :param preload_path:
        :return:
        """
        # init data
        train_loss = []
        self._show_init_data()
        num_validate_per_batch = num_validate_per_batch if num_validate_per_batch != -1 else len(self.training_data) - 1
        num_show_log = num_show_log if num_show_log != -1 else len(self.training_data) -1

        # setting
        self.model.train()

        ###### training ######
        val_loss = 0.0
        for epoch_index in range(num_epoch):
            pbar = tqdm(enumerate(self.training_data),total= len(self.training_data), smoothing=0.01, ncols=0)
            for batch_id, data in pbar:
                train_loss.append(self._train_per_batch(data))

                # log data
                if batch_id % num_show_log == 0 and batch_id != 0:
                    self._show_log(train_loss[-num_show_log:], val_loss, pbar)

                # validate and log
                if batch_id % num_validate_per_batch == 0 and batch_id != 0:
                    # run validation part
                    val_loss = self._validate()
                    self._check_overfit(train_loss, val_loss)
                    train_loss = []
                    self._save_model(val_loss, epoch_index, batch_id)

                if self.current_delay_overfit > 10:
                    break

    def _train_per_batch(self, in_data, target):
        """
            private func for batch train
        :param data:
        :param label:
        :return:
        """
        # transform data #
        in_data = transform_torch_vars(in_data, self.is_cuda)
        target = transform_torch_vars(target, self.is_cuda)

        # cal loss and optimize
        self.optimizer.zero_grad()
        output = self.model.forward(in_data)
        loss = self.loss_func(output, target)
        current_loss = loss.item()

        if self.is_apex:
            # with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            pass
        else:
            loss.backward()
        self.optimizer.step()

        return current_loss

    def _validate(self):
        """
        private validate func
        :return: average of validated loss
        """
        self.model.eval()
        loss_val = 0
        with torch.no_grad():
            for data_1, data_2 in tqdm(self.validate_data):
                ### transform data ###
                in_data = transform_torch_vars(data_1, self.is_cuda)
                target = transform_torch_vars(data_2, self.is_cuda)

                # cal loss
                output = self.model.forward(in_data)
                temp_loss = self.loss_func(output, target)
                loss_val += temp_loss.item()

        self.model.train()

        return loss_val / len(self.validate_data)

##################### customable #############

    def _show_init_data(self):
        in_img = next(iter(self.training_data))
        # for i in range(2):
        #     grid_in = torchvision.utils.make_grid(in_img[i])
        #     self.writer.add_image("images", grid_in, i)

    def _show_log(self, train_info, validate_info,pbar):
        self.writer.add_scalars("loss", {
            "Loss/train": train_info,
            "Loss/val": validate_info
        })
        pbar.set_postfix(loss=train_info, loss_val=validate_info)

    def _check_overfit(self, train_loss, val_loss):
        # check overfit.
        if self.best_current_loss <= val_loss and val_loss >= mean(train_loss) + GAP:
            self.current_delay_overfit += 1
            print("overfit delay %d" % self.current_delay_overfit)
        else:
            self.current_delay_overfit = 0

    def _save_model(self, val_loss, epoch_index, batch_id):
        # early stop.
        if self.best_current_loss > val_loss:
            self.best_current_loss = val_loss
            print("save the best _model \t new best loss \t %.5f to file %s" % (self.best_current_loss,self.checkpoint_path))
            if self.is_cuda:
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                torch.save(self.model.state_dict(), self.checkpoint_path)


class EvaluationProcess:
    def __init__(self, test_dataloader, model, pretrained_model_path, evaluate_metric = None, gpus = None):
        """
            evaluate _model
        :param test_dataloader:
        :param model: predefined _model
        :param pretrained_model_path: path to pretrain _model
        :param gpus: list of GPUs will be used (default None -> CPU)
        """
        self.test_data = test_dataloader
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.writer = SummaryWriter()
        self.evaluate_metric = evaluate_metric

        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.model.load_state_dict(torch.load(self.pretrained_model_path))
        self.gpus, self.model = setting_cuda(gpus, self.model)
        self.is_cuda = len(self.gpus) >= 1

        self.result = {}

    def _show_init_data(self):
        in_img = next(iter(self.test_data))
        grid_in = torchvision.utils.make_grid(in_img[0])
        self.writer.add_image("images", grid_in)

    def test(self, num_show = 1):
        target = []
        predict = []
        predict_without_threshold = []

        self.model.eval()
        losssum = 0
        for (in_data, out_target) in tqdm(self.test_data):
            ### transform data ###
            in_data = transform_torch_vars(in_data, self.is_cuda)
            out_target = transform_torch_vars(out_target, self.is_cuda)

            # cal loss
            output = self.model.forward(in_data)
            # temp_loss = self.evaluate_metric(output, out_target)
            # losssum += temp_loss.item()

            self._show_result(output, out_target)
            # out_pred = self._postprocess_output(output)
            # predict.extend(self._postprocess_label(out_pred))
            # predict_without_threshold.extend(self._postprocess_label_without_threshold(out_pred))
            # target.extend(self._postprocess_label_without_threshold(out_target))

            num_show -= 1
            if num_show <= 0:
                break

        print(losssum/len(self.test_data))

        # FileTool.writePickle("predict.pkl", predict_without_threshold)
        # self._print_out_result(target, predict)

    def _show_result(self, output, target):
        grid_in = torchvision.utils.make_grid(output[4])
        self.writer.add_image("images_reconstruct", grid_in)
        grid_in = torchvision.utils.make_grid(target)
        self.writer.add_image("images_target", grid_in)

    def _evaluation_func(self, pred, target):
        """
            evaluate base on pred and target and save them on result variable
        :param pred:
        :param target:
        :return: None
        """
        pass


    def _print_out_result(self, target, pred):
        """
        change code here to display result
        :param pred:
        :param target:
        :return:
        """
        FileTool.writePickle("target.pkl", target)
        FileTool.writePickle("predict.pkl", pred)

        print(precision_recall_fscore_support(target, pred))
        print(accuracy_score(target, pred))

    def _postprocess_output(self, out):
        """
        change code here for each output post process
        :param out:
        :return:
        """
        return torch.sigmoid(out)

    def _postprocess_label(self, out):
        """
        change code here for each process
        :param out:
        :return:
        """
        temp = list(out.detach().cpu().numpy().flatten())
        temp = list(map(lambda x: 1.0 if x >= 0.5 else 0.0, temp))

        return temp

    def _postprocess_label_without_threshold(self, out):
        """
        change code here for each process
        :param out:
        :return:
        """
        temp = list(out.detach().cpu().numpy().flatten())

        return temp


class EncoderProcess:
    def __init__(self, test_dataloader, model, pretrained_model_path, gpus = None):
        """
            evaluate _model
        :param test_dataloader:
        :param model: predefined _model
        :param pretrained_model_path: path to pretrain _model
        :param gpus: list of GPUs will be used (default None -> CPU)
        """
        self.test_data = test_dataloader
        self.model = model
        self.pretrained_model_path = pretrained_model_path

        self.model.load_state_dict(torch.load(self.pretrained_model_path))
        self.gpus, self.model = setting_cuda(gpus, self.model)
        self.is_cuda = len(self.gpus) >= 1

        self.result = {}

    def encode(self):
        encoder_file = []

        self.model.eval()
        for (in_data, out_target) in tqdm(self.test_data):
            ### transform data ###
            in_data = transform_torch_vars(in_data, self.is_cuda)
            out_data = transform_torch_vars(out_target, self.is_cuda)

            # cal loss
            output = self.model.forward_encode(in_data[0])
            predict = list(output.detach().cpu().numpy())
            target = self._postprocess_label_without_threshold(out_data)
            encoder_file.extend([[data, target[id]] for id, data in enumerate(predict)])

        FileTool.writePickle("features_siamese.pkl", encoder_file)

    def _postprocess_label_without_threshold(self, out):
        """
        change code here for each process
        :param out:
        :return:
        """
        temp = list(out.detach().cpu().numpy().flatten())

        return temp


class ProcessData:
    def __init__(self, init_dataloader, model, pretrained_model_path, post_process_fn = None, gpus = None,
                 with_support_output = False):
        """
            evaluate _model
        :param init_dataloader:
        :param model: predefined _model
        :param pretrained_model_path: path to pretrain _model
        :param gpus: list of GPUs will be used (default None -> CPU)
        :param with_support_output: output need infor of input to export result
        :param post_process_fn: post_process
        """
        self.init_data = init_dataloader
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.writer = SummaryWriter()

        self.model.load_state_dict(torch.load(self.pretrained_model_path))
        self.gpus, self.model = setting_cuda(gpus, self.model)
        self.is_cuda = len(self.gpus) >= 1

        self.result = {}
        self.post_func = post_process_fn
        self.with_support_output = with_support_output

    def run(self):
        self.model.eval()
        self.result = [[],[]]
        for ind,in_data in tqdm(enumerate(self.init_data), total=len(self.init_data)):
            process_data = in_data[0] if self.with_support_output else in_data
            support_output = in_data[1] if self.with_support_output else None

            process_data = transform_torch_vars(process_data, self.is_cuda)
            out_pred = self.model.forward(process_data)

            if self.post_func is not None:
                self.post_func(out_pred, support_output)

        self.post_func.finish_process()

        return self.result


class EvaluationEncoderProcess:
    def __init__(self, test_data, model, pretrained_model_path, evaluate_metric = None, gpus = None):
        """
            evaluate _model
        :param test_dataloader:
        :param model: predefined _model
        :param pretrained_model_path: path to pretrain _model
        :param gpus: list of GPUs will be used (default None -> CPU)
        """
        self.test_data = test_data
        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.evaluate_metric = evaluate_metric

        self.model.load_state_dict(torch.load(self.pretrained_model_path))
        self.gpus, self.model = setting_cuda(gpus, self.model)
        self.is_cuda = len(self.gpus) >= 1

        self.result = {}

    def test(self):
        target = []
        predict = []
        predict_without_threshold = []

        self.model.eval()
        losssum = 0
        for (in_data, out_target) in tqdm(self.test_data):
            ### transform data ###
            in_data_1 = transform_torch_vars(in_data[0], self.is_cuda)
            in_data_2 = transform_torch_vars(in_data[1], self.is_cuda)
            out_data = transform_torch_vars(out_target, self.is_cuda)

            # cal loss
            output = self.model.forward_encoder_vector(in_data_1, in_data_2)
            out_data = torch.unsqueeze(out_data,1)
            temp_loss = self.evaluate_metric(output, out_data)
            losssum += temp_loss.item()

            out_pred = self._postprocess_output(output)
            predict.extend(self._postprocess_label(out_pred))
            predict_without_threshold.extend(self._postprocess_label_without_threshold(out_pred))
            target.extend(self._postprocess_label_without_threshold(out_target))

        print(losssum/len(self.test_data))

        FileTool.writePickle("predict_score.pkl", predict_without_threshold)
        self._print_out_result(target, predict)

    def _evaluation_func(self, pred, target):
        """
            evaluate base on pred and target and save them on result variable
        :param pred:
        :param target:
        :return: None
        """
        pass


    def _print_out_result(self, target, pred):
        """
        change code here to display result
        :param pred:
        :param target:
        :return:
        """
        FileTool.writePickle("target.pkl", target)
        FileTool.writePickle("predict.pkl", pred)

        print(precision_recall_fscore_support(target, pred))
        print(accuracy_score(target, pred))

    def _postprocess_output(self, out):
        """
        change code here for each output post process
        :param out:
        :return:
        """
        return torch.sigmoid(out)

    def _postprocess_label(self, out):
        """
        change code here for each process
        :param out:
        :return:
        """
        temp = list(out.detach().cpu().numpy().flatten())
        temp = list(map(lambda x: 1.0 if x >= 0.5 else 0.0, temp))

        return temp

    def _postprocess_label_without_threshold(self, out):
        """
        change code here for each process
        :param out:
        :return:
        """
        temp = list(out.detach().cpu().numpy().flatten())

        return temp






