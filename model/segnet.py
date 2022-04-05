from torch import nn
from backbone import densenet121, densenet169, densenet201, densenet161
import torch.nn.functional as F
import torch


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock_(nn.Module):
    def __init__(self, up_in1, up_in2, up_out):
        super().__init__()
        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.x_conv_ = nn.Conv2d(up_in2, up_in1, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(up_out)
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.xavier_normal_(self.x_conv_.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):
        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        x_p = self.x_conv_(x_p)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p


class UnetBlock(nn.Module):
    def __init__(self, up_in1, up_out, size):
        super().__init__()
        self.x_conv = nn.Conv2d(up_in1, up_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(up_out)
        nn.init.xavier_normal_(self.x_conv.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, up_p, x_p):
        up_p = F.interpolate(up_p, scale_factor=2, mode='bilinear', align_corners=True)
        cat_p = torch.add(up_p, x_p)
        cat_p = self.x_conv(cat_p)
        cat_p = F.relu(self.bn(cat_p))

        return cat_p


class Extractor(nn.Module):
    def __init__(self, densenet='densenet161'):
        super().__init__()
        if densenet == 'densenet121':
            base_model = densenet121
        elif densenet == 'densenet169':
            base_model = densenet169
        elif densenet == 'densenet201':
            base_model = densenet201
        elif densenet == 'densenet161':
            base_model = densenet161
        else:
            raise Exception('The Densenet Model only accept densenet121, densenet169, densenet201 and densenet161')
        layers = list(base_model(pretrained=True).children())
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers[0]
        self.sfs = [SaveFeatures(base_layers[0][2])]
        self.sfs.append(SaveFeatures(base_layers[0][4]))
        self.sfs.append(SaveFeatures(base_layers[0][6]))
        self.sfs.append(SaveFeatures(base_layers[0][8]))
        self.up1 = UnetBlock_(2208, 2112, 768)
        self.up2 = UnetBlock(768, 384, 768)
        self.up3 = UnetBlock(384, 256, 384)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)

        return x

    def close(self):
        for sf in self.sfs: sf.remove()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)

    def forward(self, x, dropout=True):
        x = self.conv1(x)
        if dropout:
            x = F.dropout2d(x, p=0.3)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)

        return x


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, downsample=True):
        super(Projector, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.conv1 = nn.Conv2d(self.in_dim, self.in_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x


if __name__ == "__main__":
    extractor = Extractor()
    classifier = Classifier()
    projector = Projector(256, 128)
    extractor.eval()
    classifier.eval()
    projector.eval()
    input = torch.rand(1, 3, 320, 320)
    a = extractor(input)
    b = classifier(a)
    c = projector(a)
