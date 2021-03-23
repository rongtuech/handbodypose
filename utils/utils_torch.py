from torch.autograd import Variable
import torch
import os
import argparse

def parse_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(prog='vae params',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description="in and out folder")
    parser.add_argument('-s', '--state', type=str, default="train",
                        help=r' 3 state train, evaluation, process')
    parser.add_argument('-i', '--input', type=str, default=os.path.join('.', 'input'),
                        help=r'directory with input images (default: "%(default)s")')
    parser.add_argument('-o', '--output', type=str, default=os.path.join('.', 'output'),
                        help=r'directory for output images (default: "%(default)s")')
    parser.add_argument('-w', '--weights', type=str, default=None,
                        help=r'path to U-net weights (default: "%(default)s")')
    parser.add_argument('-w2', '--weights2', type=str, default=None,
                        help=r'path to U-net weights (default: "%(default)s")')
    parser.add_argument('-b', '--batchsize', type=int, default=20,
                        help=r'number of images, simultaneously sent to the GPU (default: %(default)s)')
    parser.add_argument('-g', '--gpus', type=str, default="0",
                        help=r'number of GPUs for binarization (default: %(default)s)')
    parser.add_argument('-tr', '--train', type=str, help =r'path to train folder')
    parser.add_argument('-v', '--val', type=str, help =r'path to validate folder')
    parser.add_argument('-t', '--test', type=str, help =r'path to test folder')
    parser.add_argument('-r', '--lr', type = float, default=0.00005)
    parser.add_argument('-m', '--model', type= str, default = "simple")
    parser.add_argument('-c', '--cluster', type=int, default=32)
    parser.add_argument("-e", "--num_encode", type=int, default=64)
    return parser.parse_args()

def transform_torch_vars(in_data, is_gpu = False):
    if type(in_data) is list:
        for i in range(len(in_data)):
            if in_data[i] is not None:
                in_data[i] = Variable(in_data[i].cuda()) if is_gpu else Variable(in_data[i])
            else:
                in_data[i] = None
    else:
        in_data = Variable(in_data.cuda()) if is_gpu else Variable(in_data)
    return in_data

def transform_torch_targets(in_data, is_gpu = False):
    if is_gpu:
        if type(in_data) is list:
            for i in range(len(in_data)):
                in_data[i] = in_data[i].cuda()
        else:
            in_data = in_data.cuda()
    return in_data

def setting_cuda(gpus, model):
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        gpu_list = gpus if type(gpus) is list else gpus.split(",")

        model.cuda()
        if len(gpu_list) > 1:
            model = torch.nn.DataParallel(model)

        print("Use gpu:", gpu_list, "to train.")
    else:
        gpu_list = []

    return gpu_list, model


def freezing_params_in_model(model, exception_layers = ()):
    exception_layers = list(exception_layers)
    for name, child in model.named_children():
        if name in exception_layers:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False


class CustomCollateForSiameseSequence():
    def __pad(self, data, max_len):
        """
        :param data: np.array
        :param max_len: int
        :return: padded np.array
        """
        pad_size = list(data.shape)
        pad_size[0] = max_len - data.shape[0]
        return torch.cat((data, torch.zeros(pad_size, dtype=torch.float32)), dim=0)

    def __call__(self, batch):
        x1_s_len = list(map(lambda x: x[0].shape[0], batch))
        max_len_1 = max(x1_s_len)
        x2_s_len = list(map(lambda x: x[1].shape[0], batch))
        max_len_2 = max(x2_s_len)

        batch = list(map(lambda x:
                    (self.__pad(x[0], max_len_1), self.__pad(x[1], max_len_2), x[2]), batch))

        x1_s = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        x2_s = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
        y_s = torch.tensor(list(map(lambda x: x[2], batch))).resize_((len(x2_s_len),1))
        x1_s_len = torch.tensor(x1_s_len)
        x2_s_len = torch.tensor(x2_s_len)

        return [[x1_s, x1_s_len], [x2_s, x2_s_len]], y_s


class CustomCollateForSiameseSequenceEncoder():
    def __pad(self, data, max_len):
        """
        :param data: np.array
        :param max_len: int
        :return: padded np.array
        """
        pad_size = list(data.shape)
        pad_size[0] = max_len - data.shape[0]
        return torch.cat((data, torch.zeros(pad_size, dtype=torch.float32)), dim=0)

    def __call__(self, batch):
        x1_s_len = list(map(lambda x: x[0].shape[0], batch))
        max_len_1 = max(x1_s_len)

        batch = list(map(lambda x:
                    (self.__pad(x[0], max_len_1), x[1]), batch))

        x1_s = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        y_s = torch.tensor(list(map(lambda x: x[1], batch))).resize_((len(x1_s_len),1))
        x1_s_len = torch.tensor(x1_s_len)

        return [[x1_s, x1_s_len]], y_s

class CustomCollateSimVAE():
    def _split_and_group_data(self, data):
        output = []
        for i in range(len(data[0])):
            output.append(torch.stack(list(map(lambda x: x[i], data))))
        return output

    def _split_and_ground_elementtype(self, data):
        output = []
        for i in range(len(data[0])):
            output.append(torch.LongTensor(list(map(lambda x: x[i], data))))
        return output

    def __call__(self, batch):
        user_data = list(map(lambda x: x[0], batch))
        target_data = list(map(lambda x: x[1], batch))
        labels = list(map(lambda x: [x[2]], batch))
        user_id = list(map(lambda x: x[3], batch))

        user_data = self._split_and_group_data(user_data)
        target_data = self._split_and_group_data(target_data)
        labels = torch.FloatTensor(labels)
        user_id = self._split_and_ground_elementtype(user_id)

        return user_data, target_data, labels, user_id


class CustomCollateOnlineTriplet:
    def extract_data(self, data):
        output = []
        for i in range(len(data)):
            output.extend(data[i])
        return output

    def __call__(self, batch):
        image_data = list(map(lambda x: x[0], batch))
        target_data = list(map(lambda x: x[1], batch))

        user_data = torch.stack(self.extract_data(image_data))
        target_data = torch.LongTensor(self.extract_data(target_data))

        return user_data, target_data


class CustomCollateOnlineTripletFreeSize:
    def extract_data(self, data):
        output = []
        for i in range(len(data)):
            output.extend(data[i])
        return output

    def split_bunch(self, data):
        small = []
        big = []
        for i in range(len(data)):
            if data[i].shape[2] > 300:
                big.append(data[i])
            else:
                small.append(data[i])
        return big, small

    def __call__(self, batch):
        image_data = list(map(lambda x: x[0], batch))
        target_data = list(map(lambda x: x[1], batch))

        big, small = self.split_bunch(self.extract_data(image_data))
        if len(big) <= 0:
            user_data_big = None
        else:
            user_data_big  = torch.stack(big)
        if len(small) <=0:
            user_data_small = None
        else:
            user_data_small = torch.stack(small)
        target_data = torch.LongTensor(self.extract_data(target_data))

        return [user_data_big, user_data_small], target_data



class PCATorch():
    def __call__(self, X, k, center=True, scale=False):
        X = X.float()
        n, p = X.size()
        ones = torch.ones(n).view([n, 1])
        h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
        H = torch.eye(n) - h
        X_center = torch.mm(H.double(), X.double())
        covariance = 1 / (n - 1) * torch.mm(X_center.t(), X_center).view(p, p)
        print("cal diag")
        scaling = torch.sqrt(1 / torch.diag(covariance)).double() if scale else torch.ones(p).double()
        scaled_covariance = torch.mm(torch.diag(scaling).view(p, p), covariance)
        print("cal eig")
        eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
        print(eigenvalues[:10])
        print(eigenvectors.shape)
        components = (eigenvectors[:, :k]).type(torch.FloatTensor)
        print(X.shape)
        print(components.shape)
        return X.mm(components)


def chi2_distance(A,B):
    sumAB = A + B
    subAB = A - B
    chi = 0.5 * torch.sum((subAB ** 2) / sumAB)

    return chi.item()
