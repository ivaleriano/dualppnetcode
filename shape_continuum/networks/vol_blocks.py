import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import numpy as np
from collections import OrderedDict


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_cls(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, stride=2):
        super(down_cls, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_cls(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_cls, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)

        return self.conv(x)


class fc_cls(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(fc_cls, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ELU(inplace=True),)

    def forward(self, x):
        x = self.fc(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class disc_in(nn.Module):
    def __init__(self, in_ch, out_ch, params):
        super(disc_in, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=params["pool"], stride=params["stride_pool"]),
            # nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.conv(x)


class disc_out(nn.Module):
    def __init__(self, in_ch, out_ch, params):
        super(disc_out, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=params["pool"], stride=params["stride_pool"])

    def forward(self, x):
        return self.conv(x)


class disc_down(nn.Module):
    def __init__(self, in_ch, out_ch, params):
        super(disc_down, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=params["pool"], stride=params["stride_pool"]),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, params):
        super(down, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=params["pool"], stride=params["stride_pool"])

    def forward(self, x):
        pre_pool = self.conv(x)

        return self.pool(pre_pool), pre_pool


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class up_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_bn, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1):
        x = self.up(x1)

        return self.conv(x)


class bottle_neck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(bottle_neck, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.softmax = F.log_softmax()

    def forward(self, x):
        x = self.conv(x)
        return x  # F.softmax(x)


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class ConvBnReLU_old(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ConvBnReLU(nn.Module):

    def __init__(self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                     kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock_old(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FilmBlock(nn.Module):

    def __init__(self,
        in_channels,
        out_channels,
        bn_momentum=0.1,
        stride=2,
        ndim_non_img=31,
        location=0,
        activation='linear',
        scale=True,
        shift=True):

        super().__init__()

        # init resblock
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum)
            )
        else:
            self.downsample = None

        # location decoding
        self.location = location
        film_dims = 0
        if location in [0, 1, 3]:
            film_dims = in_channels
        elif location in [2, 4, 5, 6, 7]:
            film_dims = out_channels
        else:
            raise ValueError(f'Invalid location specified: {location}')

        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = film_dims
            self.scale = None
            self.shift = None
            film_dims = 2*film_dims
        elif scale == False and shift == False:
            raise ValueError(f'FilmBlock must either do scale or shift')
        elif scale == False:
            self.scale = 1
            self.shift = None
        elif shift == False:
            self.shift = 0
            self.scale = None
        else:
            raise ValueError('Unkown error occured')

        # create aux net
        layers = [('aux_base', nn.Linear(ndim_non_img, 8, bias=False)),
                              ('aux_relu', nn.ReLU()),
                              ('aux_dropout', nn.Dropout(p=0.2, inplace=True)),
                              ('aux_out', nn.Linear(8, film_dims, bias=False))]
        self.aux=nn.Sequential(OrderedDict(layers))
        if activation == 'sigmoid':
            self.scale_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.scale_activation = nn.Tanh()
        elif activation == 'linear':
            self.scale_activation = None
        else:
            raise ValueError(f'Invalid input on activation {activation}')

        # sanity check
        if self.location == 2 and self.downsample is None:
            raise ValueError('this setup is equivalent to location=1 and no downsampling!')

    def rescale_features(self, feature_map, x_aux):

        attention = self.aux(x_aux)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale == None:
            v_scale = attention
            v_scale= v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift == None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
        else:
            raise Exception(f'Ooops, something went wrong: {self.scale}, {self.shift}')

        return (v_scale * feature_map) + v_shift

    def forward(self, x, x_aux):

        if self.location == 0:
            x = self.rescale_features(x, x_aux)
        
        residual = x

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            x = self.rescale_features(x, x_aux)
        out = self.conv1(x)
        out = self.bn1(out)

        if self.location == 4:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        if self.location == 5:
            out = self.rescale_features(out, x_aux)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.location == 2:
                residual = self.rescale_features(residual, x_aux)

        if self.location == 6:
            out = self.rescale_features(out, x_aux)
        out += residual

        if self.location == 7:
            out = self.rescale_features(out, x_aux)
        out = self.relu(out)

        return out


