from losses import CrossEntropy, EntropyMinimization, ConsistencyWeight, ContrastiveLoss
from segnet import Extractor, Classifier, Projector
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDCL(nn.Module):
    def __init__(self, opts):
        super(CDCL, self).__init__()
        self.mode = opts.mode
        self.ignore_index = opts.ignore_index
        self.reduction = opts.reduction
        self.sup_loss = CrossEntropy(self.reduction, self.ignore_index)

        self.extractor = Extractor()
        self.classifier = Classifier()

        if self.mode == 'semi':
            self.epoch_semi = opts.epoch_semi
            self.in_dim = opts.in_dim
            self.out_dim = opts.out_dim
            self.downsample = opts.downsample
            self.projector = Projector(self.in_dim, self.out_dim, self.downsample)

            self.capacity = opts.capacity
            self.count = opts.count
            self.feature_bank = []
            self.label_bank = []
            self.FC_bank = []
            self.patch_num = opts.patch_num
            self.bdp_threshold = opts.bdp_threshold
            self.fdp_threshold = opts.fdp_threshold

            self.weight_contr = opts.weight_contr
            self.weight_ent = opts.weight_ent
            self.weight_cons = opts.weight_cons
            self.max_epoch = opts.max_epoch
            self.ramp = opts.ramp
            self.threshold = opts.threshold
            self.contrastive_loss = ContrastiveLoss(self.bdp_threshold, self.fdp_threshold, self.temp)
            self.entropy_minization = EntropyMinimization(self.reduction)
            self.get_consistency_weight = ConsistencyWeight(self.weight_cons, self.max_epoch, self.ramp)

    def forward(self, x_l=None, y_l=None, x_ul=None, epoch=None, proj_ul_ema=None, z_ul_ema=None):

        if self.mode == 'suponly':
            fea_l = self.extractor(x_l)
            fea_l = self.classifier(fea_l)
            z_l = F.interpolate(fea_l, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(z_l, y_l)

            return loss_sup, z_l

        elif self.mode == 'semi':

            fea_l = self.extractor(x_l)
            fea_l = self.classifier(fea_l)
            z_l = F.interpolate(fea_l, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(z_l, y_l)
            loss_tot = loss_sup

            if epoch < self.epoch_semi:
                return loss_tot, z_l

            x_ul = x_ul['strong_aug']
            fea_ul = self.extractor(x_ul)
            proj_ul = self.projector(fea_ul)
            proj_ul = F.normalize(proj_ul, 2, 1)

            z_ul = self.classifier(fea_ul)
            loss_ent = self.entropy_minization(z_ul)

            max_probs, pseudo_label = torch.softmax(z_ul, 1).max(dim=1)
            max_probs_ema, pseudo_label_ema = torch.softmax(z_ul_ema, 1).max(dim=1)
            max_probs = max_probs.detach()
            max_probs_ema = max_probs_ema.detach()
            pseudo_label = pseudo_label.detach()
            pseudo_label_ema = pseudo_label_ema.detach()

            mask_confident = max_probs_ema > self.threshold
            loss_cons = self.sup_loss(z_ul, pseudo_label_ema, mask_confident)

            b, c = proj_ul.size(0), proj_ul.size(1)
            h, w = proj_ul.size(2) // self.patch_num, proj_ul.size(3) // self.patch_num
            patches = []
            patches_ema = []
            patch_labels = []
            patch_labels_ema = []
            FC = []
            FC_ema = []

            for i in range(self.patch_num * self.patch_num):
                j = i // self.patch_num
                k = i % self.patch_num
                for ii in range(b):
                    p = proj_ul[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                    p_ema = proj_ul_ema[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                    pla = pseudo_label[ii, j * h: (j + 1) * h, k * w: (k + 1) * w]
                    pla_ema = pseudo_label_ema[ii, j * h: (j + 1) * h, k * w: (k + 1) * w]
                    patches.append(p)
                    patches_ema.append(p_ema)
                    patch_labels.append(pla)
                    patch_labels_ema.append(pla_ema)
                    fc = pla.sum().item() / (h * w)
                    fc_ema = pla_ema.sum().item() / (h * w)
                    FC.append([fc] * h * w)
                    FC_ema.append([fc_ema] * h * w)

            _patches = [p.permute(1, 2, 0).contiguous().view(h * w, c) for p in patches]
            _patches_ema = [p.permute(1, 2, 0).contiguous().view(h * w, c) for p in patches_ema]
            _patches = torch.cat(_patches, 0)
            _patches_ema = torch.cat(_patches_ema, 0)
            _patch_labels = [p.contiguous().view(-1) for p in patch_labels]
            _patch_labels_ema = [p.contiguous().view(-1) for p in patch_labels_ema]
            _patch_labels = torch.cat(_patch_labels, 0)
            _patch_labels_ema = torch.cat(_patch_labels_ema, 0)
            _patches = torch.cat([_patches, _patches_ema], 0)
            _patch_labels = torch.cat([_patch_labels, _patch_labels_ema], 0)
            _FC = torch.cat([torch.tensor(FC), torch.tensor(FC_ema)], 0).view(-1).cuda()

            feature_all = _patches
            pseudo_label_all = _patch_labels
            FC_all = _FC

            self.feature_bank.append(feature_all)
            self.label_bank.append(pseudo_label_all)
            self.FC_bank.append(FC_all)
            if self.count > self.capacity:
                self.feature_bank = self.feature_bank[1:]
                self.label_bank = self.label_bank[1:]
                self.FC_bank = self.FC_bank[1:]
            else:
                self.count += 1
            feature_all = torch.cat(self.feature_bank, 0)
            pseudo_label_all = torch.cat(self.label_bank, 0)
            FC_all = torch.cat(self.FC_bank, 0)

            loss_contr_sum = 0.0
            loss_contr_count = 0

            for i_patch in range(b * self.patch_num * self.patch_num):
                patch_i = patches[i_patch]
                patch_ema_i = patches_ema[i_patch]
                pseudo_label_i = patch_labels[i_patch]
                pseudo_label_i_ema = patch_labels_ema[i_patch]

                patch_i = patch_i.permute(1, 2, 0).contiguous().view(-1, proj_ul.size(1))
                patch_i_ema = patch_ema_i.permute(1, 2, 0).contiguous().view(-1, proj_ul_ema.size(1))
                pseudo_label_i = pseudo_label_i.contiguous().view(-1)
                pseudo_label_i_ema = pseudo_label_i_ema.contiguous().view(-1)

                fc = pseudo_label_i.sum().item() / pseudo_label_i.size(0)
                if fc > self.bdp_threshold and fc < self.fdp_threshold:
                    continue
                else:
                    loss_contr_count += 1
                    FC_i = [fc] * h * w
                    FC_i = torch.tensor(FC_i).cuda()

                loss_contr = torch.utils.checkpoint.checkpoint(self.contrastive_loss, patch_i, patch_i_ema,
                                                               feature_all, pseudo_label_i,
                                                               pseudo_label_all, FC_i, FC_all)

                loss_contr = loss_contr.mean()
                loss_contr_sum += loss_contr

            loss_contr = loss_contr_sum / loss_contr_count

            loss_tot = loss_tot + self.weight_contr * loss_contr + self.get_consistency_weight(
                epoch) * loss_cons + self.weight_ent * loss_ent

            return loss_tot, z_ul
