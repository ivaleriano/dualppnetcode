import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_continuum.networks.point_utils import (
    farthest_point_sample,
    index_points,
    query_ball_point,
    sample_and_group,
    sample_and_group_all,
    square_distance,
)

# ----------PointNet Blocks ----------------


class STN3d(nn.Module):
    def __init__(self, num_points=1024, max_pooling=False):
        super(STN3d, self).__init__()
        self.max_pooling = max_pooling
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.mp1(x)
        # print(x.size())
        x, _ = torch.max(x, 2)
        # print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #
        iden = (
            torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(1, 9).repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class Encoder_2f(nn.Module):
    def __init__(self, num_points=1024, num_feats=5, global_feat=True, trans=True, batch_norm=True):
        super(Encoder_2f, self).__init__()
        self.batch_norm = batch_norm
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, num_feats, 1)

        if self.batch_norm:
            self.norm1 = nn.BatchNorm1d(64)
            self.norm2 = nn.BatchNorm1d(128)
            self.norm3 = nn.BatchNorm1d(num_feats)
        else:
            self.norm1 = nn.InstanceNorm1d(64)
            self.norm2 = nn.InstanceNorm1d(128)
            self.norm3 = nn.InstanceNorm1d(num_feats)
        self.trans = trans
        self.lat_feats = num_feats

        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.norm1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.lat_feats)
        if self.trans:
            if self.global_feat:
                return x
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x


# ----- POINTNET++ BLOCKS -----#
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, returnidx=False):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            if returnidx:
                new_xyz, new_points, centroids, corr_idx = sample_and_group(
                    self.npoint, self.radius, self.nsample, xyz, points, returnidx=True
                )
            else:
                new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points, max_indices = torch.max(new_points, 2)  # [0]
        new_xyz = new_xyz.permute(0, 2, 1)
        if returnidx:
            return new_xyz, new_points, corr_idx
        else:
            return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points, max_indices = torch.max(grouped_points, 2)  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


# ----RE-CALIBRATION BLOCKS --------------


class PointNetcSE(nn.Module):
    def __init__(self, in_channel, r=2, fc=True):
        super(PointNetcSE, self).__init__()
        num_channels_reduced = in_channel // r
        self.reduction_ratio = in_channel
        self.fc = fc
        if fc:
            self.fc1 = nn.Linear(in_channel, num_channels_reduced, bias=True)
            self.fc2 = nn.Linear(num_channels_reduced, in_channel, bias=True)
        else:
            print("Convolution instead of fully-connected!")
            self.fc1 = nn.Conv1d(in_channel, num_channels_reduced, 1)
            self.fc2 = nn.Conv1d(num_channels_reduced, in_channel, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, fc=True):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """

        # B, C, N = points.shape
        mask = torch.mean(points, 2)
        if not self.fc:
            mask = torch.unsqueeze(mask, 2)
        mask = self.relu(self.fc1(mask))
        mask = torch.sigmoid(self.fc2(mask))
        if not self.fc:
            mask = torch.squeeze(mask)
        a, b = mask.size()
        out_points = torch.mul(points, mask.view(a, b, 1))
        return out_points


class PointNetsSE(nn.Module):
    def __init__(self, in_channel, n_points, rec=False, r=2):
        super(PointNetsSE, self).__init__()
        self.conv = nn.Conv1d(in_channel, 1, 1)
        self.rec = rec
        self.n_points = n_points
        if rec:
            n_points_reduced = int(n_points / r)
            self.fc1 = nn.Linear(n_points, n_points_reduced, bias=True)
            self.fc2 = nn.Linear(n_points_reduced, n_points, bias=True)
            self.relu = nn.ReLU()

    def forward(self, points):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        B, C, N = points.size()
        mask = self.conv(points)
        if self.rec:
            mask = self.fc2(self.relu(self.fc1(mask)))
        mask = torch.sigmoid(mask)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        mask = mask.view(B, 1, N)
        out_points = torch.mul(points, mask)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return out_points


class PointNetcsSE(nn.Module):
    def __init__(self, in_channel, n_points, r=2, rec=False):
        super(PointNetcsSE, self).__init__()
        self.cSE = PointNetcSE(in_channel, r)
        self.sSE = PointNetsSE(in_channel, n_points, rec, r)

    def forward(self, points):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        output_points = torch.max(self.cSE(points), self.sSE(points))
        return output_points
