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


class ConvBnReLU(nn.Module):
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


class ResBlock(nn.Module):
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
