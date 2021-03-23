import torch.nn.functional as F
import numpy as np
import torch


def compute_loss(pafs_pred, heatmaps_pred, pafs_t, heatmaps_t):
    # light pose have only one refinement -> no need loop and ignore
    # compute loss on each stage
    if pafs_pred.shape != pafs_t.shape:
        with torch.no_grad():
            pafs_t = F.interpolate(pafs_t, pafs_pred.shape[2:], mode='bilinear', align_corners=True)
            heatmaps_t = F.interpolate(heatmaps_t, heatmaps_pred.shape[2:], mode='bilinear',
                                             align_corners=True)

    pafs_loss = torch.nn.MSELoss()(pafs_pred, pafs_t)
    heatmaps_loss = torch.nn.MSELoss()(heatmaps_pred, heatmaps_t)
    total_loss = pafs_loss + heatmaps_loss

    return total_loss, pafs_loss.item(), heatmaps_loss.item()

