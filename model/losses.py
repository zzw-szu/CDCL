import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.extractor.parameters(), model.extractor.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(ema_model.projector.parameters(), model.projector.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(ema_model.classifier.parameters(), model.classifier.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class CrossEntropy(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        _targets = targets.clone()
        if mask is not None:
            _targets[mask] = self.ignore_index

        loss = F.cross_entropy(inputs, _targets, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


class EntropyMinimization(nn.Module):
    def __init__(self, reduction='mean'):
        super(EntropyMinimization, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):
        P = torch.softmax(inputs, dim=1)
        logP = torch.log_softmax(inputs, dim=1)
        PlogP = P * logP
        loss_ent = -1.0 * PlogP.sum(dim=1)
        if self.reduction == 'mean':
            loss_ent = torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = torch.sum(loss_ent)
        else:
            pass

        return loss_ent


class ContrastiveLoss(nn.Module):
    def __init__(self, bdp_threshold, fdp_threshold, temp=0.1, eps=1e-8):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.eps = eps
        self.bdp_threshold = bdp_threshold
        self.fdp_threshold = fdp_threshold

    def forward(self, anchor, pos_pair, neg_pair, pseudo_label, pseudo_label_all, FC, FC_all):
        pos = torch.div(torch.mul(anchor, pos_pair), self.temp).sum(-1, keepdim=True)

        mask_pixel_filter = (pseudo_label.unsqueeze(-1) != pseudo_label_all.unsqueeze(0)).float()
        mask_patch_filter = (
                (FC.unsqueeze(-1) + FC_all.unsqueeze(0)) <= (self.bdp_threshold + self.fdp_threshold)).float()
        mask_pixel_filter = torch.cat([torch.ones(mask_pixel_filter.size(0), 1).float().cuda(), mask_pixel_filter], 1)
        mask_patch_filter = torch.cat([torch.ones(mask_patch_filter.size(0), 1).float().cuda(), mask_patch_filter], 1)

        neg = torch.div(torch.matmul(anchor, neg_pair.T), self.temp)
        neg = torch.cat([pos, neg], 1)
        max = torch.max(neg, 1, keepdim=True)[0]
        exp_neg = (torch.exp(neg - max) * mask_pixel_filter * mask_patch_filter).sum(-1)

        loss = torch.exp(pos - max).squeeze(-1) / (exp_neg + self.eps)
        loss = -torch.log(loss + self.eps)

        return loss


class ConsistencyWeight(nn.Module):
    def __init__(self, max_weight, max_epoch, ramp='sigmoid'):
        super(ConsistencyWeight, self).__init__()
        self.max_weight = max_weight
        self.max_epoch = max_epoch
        self.ramp = ramp

    def forward(self, epoch):
        current = np.clip(epoch, 0.0, self.max_epoch)
        phase = 1.0 - current / self.max_epoch
        if self.ramp == 'sigmoid':
            ramps = float(np.exp(-5.0 * phase * phase))
        elif self.ramp == 'log':
            ramps = float(1 - np.exp(-5.0 * current / self.max_epoch))
        elif self.ramp == 'exp':
            ramps = float(np.exp(5.0 * (current / self.max_epoch - 1)))
        else:
            ramps = 1.0

        consistency_weight = self.max_weight * ramps
        return consistency_weight
