import torch.nn as nn
import torch.nn.functional as F
from shape_continuum.networks.vol_blocks import down_cls,fc_cls,ConvBnReLU,ResBlock


class Vol_classifier(nn.Module):
    # volume classifier from Biffi et al. - Explainable Shape Analysis
    def __init__(self, opt, ncf=8):
        super(Vol_classifier, self).__init__()

        self.down0 = down_cls(opt.in_channels, ncf)
        self.down1 = down_cls(ncf, ncf * 2)
        self.down2 = down_cls(ncf * 2, ncf * 4)
        self.down3 = down_cls(ncf * 4, ncf * 8)
        self.down4 = down_cls(ncf * 8, 2, stride=1)
        self.fc0 = fc_cls(128, 128)
        self.fc1 = fc_cls(128, 96)
        self.fc2 = fc_cls(96, 48)
        self.fc3 = fc_cls(48, 24)
        self.fc4 = fc_cls(24, 12)
        self.outc = nn.Linear(12, opt.num_classes)
        self.task = opt.task

    def forward(self, input):
        bs = input.size()[0]
        d0 = self.down0(input)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        z = d4.view(bs, -1)
        c0 = self.fc0(z)
        c1 = self.fc1(c0)
        c2 = self.fc2(c1)
        c3 = self.fc3(c2)
        c4 = self.fc4(c3)
        out = self.outc(c4)
        if self.task == "clf":
            out = F.log_softmax(out, dim=1)
        return {"pred": out}


class ResNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.conv1 = ConvBnReLU(opt.in_channels, 32)
        self.pool1 = nn.MaxPool3d(2, stride=2, padding=1)  # 32
        self.block1 = ResBlock(32, 32)
        self.pool2 = nn.MaxPool3d(2, stride=2, padding=1)  # 16
        self.block2 = ResBlock(32, 64)
        self.block3 = ResBlock(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.fc = nn.Linear(128, opt.num_classes)
        self.task = opt.task

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.pool2(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        if self.task == "clf":
            out = F.log_softmax(out, dim=1)
        return {"pred": out}
