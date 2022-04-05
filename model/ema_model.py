from losses import CrossEntropy
from segnet import Extractor, Classifier, Projector
import torch.nn as nn
import torch.nn.functional as F


class CDCL_ema(nn.Module):
    def __init__(self, opts):
        super(CDCL_ema, self).__init__()
        self.mode = opts.mode
        self.ignore_index = opts.ignore_index
        self.reduction = opts.reduction
        self.sup_loss = CrossEntropy(self.reduction, self.ignore_index)

        self.extractor = Extractor()
        self.classifier = Classifier()

        if self.mode == 'semi':
            self.in_dim = opts.in_dim
            self.out_dim = opts.out_dim
            self.downsample = opts.downsample
            self.projector = Projector(self.in_dim, self.out_dim, self.downsample)

    def forward(self, x_l=None, x_ul=None):
        if self.mode == 'test':
            fea_l = self.encoder(x_l)
            fea_l = self.classifier(fea_l)
            z_l = F.interpolate(fea_l, size=x_l.size()[2:], mode='bilinear', align_corners=True)

            return z_l

        elif self.mode == 'semi':
            x_ul_ema = x_ul['weak_aug']
            fea_ul_ema = self.encoder(x_ul_ema)
            proj_ul_ema = self.projector(fea_ul_ema)
            proj_ul_ema = F.normalize(proj_ul_ema, 2, 1)
            z_ul_ema = self.classifier(fea_ul_ema)

            return proj_ul_ema, z_ul_ema
