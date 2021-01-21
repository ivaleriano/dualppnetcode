from typing import Sequence

import torch.nn as nn

from ..models.base import BaseModel
from .vol_blocks import ConvBnReLU, ResBlock, down_cls, fc_cls


class Vol_classifier(BaseModel):
    # volume classifier from Biffi et al. - Explainable Shape Analysis
    def __init__(self, in_channels: int, num_outputs: int, ncf: int = 8) -> None:
        super(Vol_classifier, self).__init__()

        self.down0 = down_cls(in_channels, ncf)
        self.down1 = down_cls(ncf, ncf * 2)
        self.down2 = down_cls(ncf * 2, ncf * 4)
        self.down3 = down_cls(ncf * 4, ncf * 8)
        self.down4 = down_cls(ncf * 8, 2, stride=1)
        self.fc0 = fc_cls(128, 128)
        self.fc1 = fc_cls(128, 96)
        self.fc2 = fc_cls(96, 48)
        self.fc3 = fc_cls(48, 24)
        self.fc4 = fc_cls(24, 12)
        self.outc = nn.Linear(12, num_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image):
        bs = image.size()[0]
        d0 = self.down0(image)
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

        return {"logits": out}


class ResNet_old(BaseModel):
    def __init__(self, in_channels: int, num_outputs: int) -> None:
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2, stride=2, padding=1)  # 32
        self.block1 = ResBlock(32, 32)
        self.pool2 = nn.MaxPool3d(2, stride=2, padding=1)  # 16
        self.block2 = ResBlock(32, 64)
        self.block3 = ResBlock(64, 128)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.fc = nn.Linear(128, num_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.pool2(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)

        return {"logits": out}


class ResNet(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.05, n_basefilters=32):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        # self.block4 = ResBlock(4*n_basefilters, 8*n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}
