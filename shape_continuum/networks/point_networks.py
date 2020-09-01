from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F

from ..models.base import BaseModel
from .point_blocks import Encoder_2f, PointNetSetAbstraction, PointNetSetAbstractionMsg


class PointNet(BaseModel):
    def __init__(self, num_points: int, num_outputs: int, batch_norm: bool, feature_transform=False) -> None:
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform
        self.encoder = Encoder_2f(num_points=num_points, num_feats=1024, batch_norm=batch_norm)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_outputs)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    @property
    def input_names(self) -> Sequence[str]:
        return ("pointcloud",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, pointcloud):
        x = self.encoder(pointcloud)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        outputs = {"logits": x}
        return outputs


class PointNet2ClsMsg(nn.Module):
    def __init__(self, num_outputs: int) -> None:
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [16, 32, 128], 0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("pointcloud",)

    @property
    def output_names(self) -> Sequence[str]:
        return (
            "logits",
            "l3_points",
        )

    def forward(self, pointcloud):
        B, _, _ = pointcloud.shape
        l1_xyz, l1_points = self.sa1(pointcloud, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        outputs = {"logits": x, "l3_points": l3_points}
        return outputs


class PointNet2ClsSsg(BaseModel):
    def __init__(self, num_outputs: int) -> None:
        super(PointNet2ClsSsg, self).__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("pointcloud",)

    @property
    def output_names(self) -> Sequence[str]:
        return (
            "logits",
            "ln_points",
        )

    def forward(self, pointcloud):
        B, _, _ = pointcloud.shape
        l1_xyz, l1_points = self.sa1(pointcloud, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        outputs = {"logits": x, "ln_points": [l1_points, l2_points, l3_points]}
        return outputs
