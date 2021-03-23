import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F


def get_manhattan_dis(input_1, input_2):
    return torch.abs(input_1 - input_2)


def get_euclid_dis(input_1, input_2):
    return torch.sqrt(torch.pow(input_1 - input_2,2))


def get_kld_normal(mu, logvar, reduction = "sum"):
    if reduction == "sum":
        return torch.sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    else:
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))


def get_kld(mu, logvar, target_mu, target_logvar, reduction = "sum"):
    if reduction == "sum":
        return torch.sum(0.5*torch.sum(target_logvar - logvar - 1 + (logvar.exp() + (mu - target_mu).pow(2)) / target_logvar.exp()))
    else:
        return torch.mean(0.5*torch.sum(target_logvar - logvar - 1 + (logvar.exp() + (mu - target_mu).pow(2)) / target_logvar.exp()))



def _pairwise_distance_p2(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _pairwise_distance_p1(x):
    # Compute the 2D matrix of distances between all the embeddings.
    # in mahattan distance
    all_pair = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    distances = all_pair.sum(dim=2)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(labels.device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2).type(dtype = torch.long)
    i_equal_k = torch.unsqueeze(label_equal, 1).type(dtype = torch.long)
    valid_labels = i_equal_j * (i_equal_k ^ 1)
    valid_labels = valid_labels.type(dtype = torch.float)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask


class DiceCoeff(nn.Module):
    def __init__(self, eps= 0.0001):
        super(DiceCoeff, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        if input.is_cuda:
            s = torch.FloatTensor(1).cuda().zero_()
        else:
            s = torch.FloatTensor(1).zero_()

        for i, c in enumerate(zip(input, target)):
            s = s + self.dice_coeff(c[0], c[1])

        num_batch = list(input.size())[0]

        return s / num_batch

    def dice_coeff(self, input, target):
        input = F.sigmoid(input)
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + self.eps

        t = (2 * self.inter.float() + self.eps) / self.union.float()
        return 1 - t


class TripletLoss(nn.Module):
    def __init__(self, alpha = 0.5):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output_sim, output_diff):
        return torch.sum(torch.max(output_sim - output_diff + self.alpha, torch.zeros_like(output_sim)))


class TripletLossNew(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLossNew, self).__init__()
        self.margin = margin

    def forward(self, distance_positive, distance_negative, size_average=True):
        losses = distance_positive - distance_negative + self.margin
        # return losses.mean() if size_average else losses.sum()

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(losses)

        # Count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)
        return triplet_loss


class SlowVAELoss(nn.Module):
    """
    cal only the slow
    """
    def __init__(self, beta = 1, gap_tempo = 0.1, gap_static = 0.01, step_static = 3):
        super(SlowVAELoss, self).__init__()
        self.beta = beta
        self.gap_tempo = gap_tempo
        self.gap_static = gap_static
        self.step_static = step_static
        self.static_ratio = 0.9

        self.compare_static = None
        self.compare_tempo = None

    def _get_slow_loss(self, mu_static):
        # cal distance between latent vars of a segment, respectively with the next segment.
        slow_feature = torch.sum(nn.PairwiseDistance(p=2)(mu_static[1:], mu_static[:-1]))


        gap_static = self.gap_static * (mu_static.shape[1] - self.step_static)
        dist_static = nn.PairwiseDistance(p=2)(mu_static[self.step_static:], mu_static[:-self.step_static])
        if self.compare_static is None:
            self.compare_static = torch.zeros_like(dist_static)
        not_constant_loss = torch.sum(torch.max(torch.add(torch.neg(dist_static),gap_static), self.compare_static))

        return torch.add(self.static_ratio*slow_feature,  (1- self.static_ratio)*not_constant_loss)

    def _get_tempo_loss(self, mu_tempo):
        gap_tempo = self.gap_tempo * (mu_tempo.shape[1] - 1)
        dist_tempo = nn.PairwiseDistance(p=2)(mu_tempo[1:], mu_tempo[:-1])
        if self.compare_tempo is None:
            self.compare_tempo = torch.zeros_like(dist_tempo)
        temp_feature = torch.sum(torch.max(torch.add(torch.neg(dist_tempo),gap_tempo), self.compare_tempo))

        return temp_feature

    def forward(self, output, target):
        mu_tempo, logvar_tempo, mu_static, logvar_static, recon_x = output

        # basic loss mse and kld for latent vars.
        MSE = F.mse_loss(recon_x, target, size_average=False)
        KLD_tempo = -0.5 * torch.sum(1 + logvar_tempo - mu_tempo.pow(2) - logvar_tempo.exp())
        KLD_static = -0.5 * torch.sum(1 + logvar_static - mu_static.pow(2) - logvar_static.exp())

        # slow feature loss
        slow =None
        for mu in mu_static:
            if slow is None:
                slow = self._get_slow_loss(mu)
            else:
                slow = slow + self._get_slow_loss(mu)

        # tempo feature loss
        tempo = None
        for mu in mu_tempo:
            if tempo is None:
                tempo = self._get_tempo_loss(mu)
            else:
                tempo = tempo + self._get_tempo_loss(mu)

        sum_loss = MSE + self.beta*(KLD_tempo + KLD_static + slow+ tempo)
        # mean of each sample in batch.
        return torch.div(sum_loss, list(recon_x.size())[0])


class GroupFeatureVAELoss(nn.Module):
    def __init__(self, beta = 0.5, gap_tempo = 0.5, gap_static = 0.01, step_static = 3):
        super(GroupFeatureVAELoss, self).__init__()
        self.beta = beta
        self.gap_tempo = gap_tempo
        self.gap_static = gap_static
        self.step_static = step_static

        self.compare_static = None
        self.compare_tempo = None

    def _get_slow_loss(self, mu_static, logvar_static, target_mus, target_logvars, index):
        # cal distance between latent vars of a segment, respectively with the next segment.
        slow_feature = torch.sum(nn.PairwiseDistance(p=2)(mu_static[1:], mu_static[:-1]))

        # cal kld
        kld_static = get_kld(mu_static, logvar_static, target_mus[index], target_logvars[index])

        # # compare kld
        # sum_kld = None
        # for i in range(len(target_mus)):
        #     if sum_kld is None:
        #         sum_kld = get_kld(mu_static, logvar_static, target_mus[i], target_logvars[i])
        #     else:
        #         sum_kld= sum_kld + get_kld(mu_static, logvar_static, target_mus[i], target_logvars[i])
        # compare_loss = kld_static / sum_kld

        # return slow_feature + kld_static + compare_loss
        return slow_feature + kld_static

    def _get_tempo_loss(self, mu_tempo, logvar_tempo):
        # # make sure that temp have no static info
        # dist_tempo = nn.PairwiseDistance(p=2)(mu_tempo[1:], mu_tempo[:-1])
        # if self.compare_tempo is None:
        #     self.compare_tempo = torch.zeros_like(dist_tempo)
        #     self.compare_tempo.detach()
        # temp_feature = torch.sum(torch.max(torch.add(torch.neg(dist_tempo),self.gap_tempo), self.compare_tempo))

        # make sure that tempo have a same attribute with normal distribution.
        KLD_tempo = get_kld_normal(mu_tempo, logvar_tempo)

        # return temp_feature + KLD_tempo
        return KLD_tempo

    def forward(self, output, target):
        mu_tempo, logvar_tempo, mu_static, logvar_static, recon_x = output
        target_img, target_slow_mu, target_slow_logvar, index = target

        # basic loss mse and kld for latent vars.
        MSE = F.mse_loss(recon_x, target_img, reduction='sum')

        # slow feature loss
        slow = self._get_slow_loss(mu_static, logvar_static, target_slow_mu, target_slow_logvar, index)

        # tempo feature loss
        tempo = self._get_tempo_loss(mu_tempo, logvar_tempo)

        sum_loss = MSE + self.beta*(slow+ tempo)
        # mean of each sample in batch.
        return torch.div(sum_loss, list(recon_x.size())[0])


class SimVAELossWithSlow(nn.Module):
    def __init__(self, beta = 1, gap_tempo = 0.5, gap_static = 0.01, step_static = 3):
        super(SimVAELossWithSlow, self).__init__()
        self.beta = beta
        self.gap_tempo = gap_tempo
        self.gap_static = gap_static
        self.step_static = step_static

        self.compare_static = None
        self.compare_tempo = None
        self.tripletloss = TripletLoss()

    def _get_slow_loss(self, mu_static, logvar_static, target_mus, target_logvars, index, sim_out):
        # cal distance between latent vars of a segment, respectively with the next segment.
        slow_feature = torch.sum(nn.PairwiseDistance(p=2)(mu_static[1:], mu_static[:-1]))

        # cal kld
        kld_static = get_kld(mu_static, logvar_static, target_mus[index], target_logvars[index])

        # compare kld
        # sum_kld = None
        # for i in range(len(target_mus)):
        #     if sum_kld is None:
        #         sum_kld = get_kld(mu_static, logvar_static, target_mus[i], target_logvars[i])
        #     else:
        #         sum_kld= sum_kld + get_kld(mu_static, logvar_static, target_mus[i], target_logvars[i])
        # compare_loss = kld_static / sum_kld

        return slow_feature + kld_static + self.tripletloss(sim_out[0], sim_out[1])

    def _get_tempo_loss(self, mu_tempo, logvar_tempo):
        # make sure that temp have no static info
        dist_tempo = nn.PairwiseDistance(p=2)(mu_tempo[1:], mu_tempo[:-1])
        if self.compare_tempo is None:
            self.compare_tempo = torch.zeros_like(dist_tempo)
            self.compare_tempo.detach()
        temp_feature = torch.sum(torch.max(torch.add(torch.neg(dist_tempo),self.gap_tempo), self.compare_tempo))

        # make sure that tempo have a same attribute with normal distribution.
        KLD_tempo = get_kld_normal(mu_tempo, logvar_tempo)

        return temp_feature + KLD_tempo
        # return KLD_tempo

    def forward(self, output, target):
        mu_tempo, logvar_tempo, mu_static, logvar_static, output_sim, recon_x = output
        target_img, target_slow_mu, target_slow_logvar, index,  = target

        # basic loss mse and kld for latent vars.
        MSE = F.mse_loss(recon_x, target_img, reduction='sum')

        # slow feature loss
        slow = self._get_slow_loss(mu_static, logvar_static, target_slow_mu, target_slow_logvar, index, output_sim)

        # tempo feature loss
        tempo = self._get_tempo_loss(mu_tempo, logvar_tempo)

        sum_loss = MSE + self.beta*(slow+ tempo)
        # mean of each sample in batch.
        return torch.div(sum_loss, list(recon_x.size())[0])


class SimVAELoss(nn.Module):
    def __init__(self, gap_triplet=1, is_triplet = False):
        super(SimVAELoss, self).__init__()
        print("is triplet" + str(is_triplet))
        self.tripletloss = TripletLoss(alpha=gap_triplet)
        self.is_triplet = is_triplet

    def forward(self, output, target_img, target_siamese):
        mse = None
        kld = None
        siamese_loss = None
        for i in range(len(output[0])):
            temp_out, kld_loss = output[0][i]
            if mse is None:
                mse = F.mse_loss(temp_out, target_img[i], size_average=False)/target_img[0].size(0)
                kld = kld_loss
            else:
                mse += F.mse_loss(temp_out, target_img[i])/target_img[0].size(0)
                kld += kld_loss
        mse = mse / len(output[0])
        kld = kld / len(output[0])
        vae_loss = mse + kld

        if self.is_triplet:
            siamese_loss = self.tripletloss(output[1][0],output[1][1])
        else:
            siamese_loss = F.binary_cross_entropy_with_logits(output[1][0],
                                                              target_siamese)
        return vae_loss.unsqueeze(0)+ 100*siamese_loss.unsqueeze(0), mse, kld, siamese_loss


class RegularVAELoss(nn.Module):
    def forward(self, recon_x, target_x, mu_tempo, logvar_tempo, mu_static, logvar_static):
        MSE = F.binary_cross_entropy(recon_x, target_x, reduction='sum')
        KLD_tempo = -0.5 * torch.sum(1 + logvar_tempo - mu_tempo.pow(2) - logvar_tempo.exp())
        KLD_static = -0.5 * torch.sum(1 + logvar_static - mu_static.pow(2) - logvar_static.exp())
        return torch.div(MSE + KLD_tempo + KLD_static, list(recon_x.size())[0])


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared = False, p = 1):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared
        self.p = p

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        if self.p == 1:
            pairwise_dist = _pairwise_distance_p1(embeddings)
        else:
            pairwise_dist = _pairwise_distance_p2(embeddings, squared=self.squared)
        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)

        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)
            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin
            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask
            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)
            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)
            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


class TripletVAEClassifierLoss(nn.Module):
    def __init__(self, gap_triplet=1.0, dist_pow = 2):
        super(TripletVAEClassifierLoss, self).__init__()
        # self.tripletloss = TripletLossNew(margin=gap_triplet)
        self.tripletloss = nn.TripletMarginLoss(margin=2.0, p=1)
        self.classifier_loss = nn.CrossEntropyLoss()
        self.dist_pow = dist_pow

    def forward(self, output, target_user_ids, validate=False):
        cross_entropy_1 = None
        for i in range(len(output)):
            embedde_x, out_classifier_1 = output[i] # embedded_x, x_reconst, kl_loss, out_classifier
            if not validate:
                if cross_entropy_1 is not None:
                    cross_entropy_1 = cross_entropy_1 + self.classifier_loss(out_classifier_1, target_user_ids[i])
                else:
                    cross_entropy_1 = self.classifier_loss(out_classifier_1, target_user_ids[i])

            else:
                cross_entropy_1 = 0
                cross_entropy_2 = 0
        cross_entropy_1 = cross_entropy_1 / len(output)

        triplet_loss = self.tripletloss(output[0][0],output[1][0],output[2][0])
        losses = [cross_entropy_1, triplet_loss*10]
        final_loss = 0
        for ele in losses:
            if not torch.is_tensor(ele) or not torch.isnan(ele):
                final_loss += ele
        if final_loss == 0:
            final_loss = triplet_loss
        if validate:
            return triplet_loss
        else:
            return final_loss, triplet_loss, cross_entropy_1


class OnlineTripletClassifyLoss(nn.Module):
    def __init__(self, margin=2.0, dist_pow = 1):
        super(OnlineTripletClassifyLoss, self).__init__()
        self.tripletloss = HardTripletLoss(margin=margin, p=dist_pow)
        self.classifier_loss = nn.CrossEntropyLoss()
        self.dist_pow = dist_pow

    def forward(self, embedding_feature, pred, target_user_ids, validate=False):
        triplet_loss = self.tripletloss(embedding_feature, target_user_ids)
        if validate:
            return triplet_loss
        else:
            cross_entropy = self.classifier_loss(pred, target_user_ids)
            return cross_entropy + 5*triplet_loss, triplet_loss, cross_entropy
            # return triplet_loss, triplet_loss, cross_entropy
