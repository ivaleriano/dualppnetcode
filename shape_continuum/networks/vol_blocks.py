from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


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
    def __init__(
        self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
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
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
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


class FilmBase(nn.Module, metaclass=ABCMeta):
    """ Absract base class for models that are related to FiLM of Perez et al
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float,
        stride: int,
        ndim_non_img: int,
        location: int,
        activation: str,
        scale: bool,
        shift: bool,
    ) -> None:

        super(FilmBase, self).__init__()

        # sanity checks
        if location not in set(range(8)):
            raise ValueError(f"Invalid location specified: {location}")
        if activation not in {"tanh", "sigmoid", "linear"}:
            raise ValueError(f"Invalid location specified: {location}")
        if (not isinstance(scale, bool) or not isinstance(shift, bool)) or (not scale and not shift):
            raise ValueError(
                f"scale and shift must be of type bool:\n    -> scale value: {scale}, "
                "scale type {type(scale)}\n    -> shift value: {shift}, shift type: {type(shift)}"
            )
        # ResBlock
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None
        # Film-specific variables
        self.location = location
        if self.location == 2 and self.downsample is None:
            raise ValueError("This is equivalent to location=1 and no downsampling!")
        # location decoding
        self.film_dims = 0
        if location in {0, 1, 3}:
            self.film_dims = in_channels
        elif location in {2, 4, 5, 6, 7}:
            self.film_dims = out_channels
        if activation == "sigmoid":
            self.scale_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = nn.Tanh()
        elif activation == "linear":
            self.scale_activation = None

    @property
    @abstractmethod
    def rescale_features(self, feature_map, x_aux):
        """method to recalibrate feature map x"""

    def forward(self, feature_map, x_aux):

        if self.location == 0:
            feature_map = self.rescale_features(feature_map, x_aux)

        residual = feature_map

        if self.location == 1:
            residual = self.rescale_features(residual, x_aux)

        if self.location == 3:
            feature_map = self.rescale_features(feature_map, x_aux)
        out = self.conv1(feature_map)
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


class FilmBlock(FilmBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 14,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
    ):

        super(FilmBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None
        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 8, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_out", nn.Linear(8, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        attention = self.aux(x_aux)

        assert (attention.size(0) == feature_map.size(0)) and (
            attention.dim() == 2
        ), f"Invalid size of output tensor of auxiliary network: {attention.size()}"

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise Exception(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class ZeCatBlock(FilmBase):
    # Block for ZeCatNet
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 14,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
    ) -> None:

        super(ZeCatBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        aux_input_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img + aux_input_dims, 8, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_out", nn.Linear(8, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def rescale_features(self, feature_map, x_aux):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        if self.scale == self.shift:
            v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = attention
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = attention
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise Exception(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift


class ZeNullBlock(FilmBase):
    # Block for ZeNullNet
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 14,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
    ) -> None:

        super(ZeNullBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        squeeze_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 8)),
            ("aux_relu", nn.ReLU()),
            ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_out", nn.Linear(8, 8 * self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.squeeze = nn.Linear(squeeze_dims, 8, bias=False)

    def rescale_features(self, feature_map, x_aux):

        squeeze_vector = self.global_pool(feature_map)
        squeeze_vector = squeeze_vector.view(squeeze_vector.size(0), -1)
        squeeze_vector = self.squeeze(squeeze_vector)

        weights = self.aux(x_aux)  # matrix weights
        weights = weights.view(
            *squeeze_vector.size(), -1
        )  # squeeze_vecotr.size is (batch_size, 8). After reshaping, weights has size (batch_size, 8, film_dims)
        weights = torch.einsum("bi,bij->bj", squeeze_vector, weights)  # j = alpha and beta

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(weights, self.split_size, dim=1)
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = weights
            v_scale = v_scale.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = weights
            v_shift = v_shift.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(feature_map)
        else:
            raise Exception(f"Ooops, something went wrong: {self.scale}, {self.shift}")

        return (v_scale * feature_map) + v_shift


class ZeNewBlock(FilmBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 2,
        ndim_non_img: int = 14,
        location: int = 0,
        activation: str = "linear",
        scale: bool = True,
        shift: bool = True,
    ) -> None:

        super(ZeNewBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_momentum=bn_momentum,
            stride=stride,
            ndim_non_img=ndim_non_img,
            location=location,
            activation=activation,
            scale=scale,
            shift=shift,
        )

        squeeze_dims = self.film_dims
        # shift and scale decoding
        self.split_size = 0
        if scale and shift:
            self.split_size = self.film_dims
            self.scale = None
            self.shift = None
            self.film_dims = 2 * self.film_dims
        elif not scale:
            self.scale = 1
            self.shift = None
        elif not shift:
            self.shift = 0
            self.scale = None

        # create aux net
        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 8)),
            ("aux_relu", nn.ReLU()),
            ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_out", nn.Linear(8, 8 + self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.squeeze = nn.Linear(squeeze_dims, 8, bias=False)

    def rescale_features(self, feature_map, x_aux):

        squeeze_vector = self.global_pool(feature_map)
        squeeze_vector = squeeze_vector.view(squeeze_vector.size(0), -1)
        squeeze_vector = self.squeeze(squeeze_vector)

        low_rank = self.aux(x_aux)  # matrix weights, shape (batch_size, 8+FilmDims)
        v0, v1 = torch.split(
            low_rank, [8, low_rank.size(1) - 8], dim=1
        )  # v0 size -> (batchsize, 8), v1 size -> (batchsize, FilmDims)

        weights = torch.einsum("bi, bj->bij", v0, v1)  # weights size -> (batchsize, 8, FilmDims)
        weights = torch.einsum("bi,bij->bj", squeeze_vector, weights)  # j = alpha and beta = filmdims

        if self.scale == self.shift:
            v_scale, v_shift = torch.split(weights, self.split_size, dim=1)
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.scale is None:
            v_scale = weights
            v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
            v_shift = self.shift
            if self.scale_activation is not None:
                v_scale = self.scale_activation(v_scale)
        elif self.shift is None:
            v_scale = self.scale
            v_shift = weights
            v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        else:
            raise Exception(
                f"Sanity checking on scale and shift failed. Must be of type bool or None: {self.scale}, {self.shift}"
            )

        return (v_scale * feature_map) + v_shift
