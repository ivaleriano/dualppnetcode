"""
Code adapted from Gong et al. SpiralNet++ Pytorch implementation
 https://github.com/sw-gong/spiralnet_plus
"""


import torch
import torch.nn as nn
from .mesh_blocks import SpiralEnblock
from typing import Sequence


class SpiralNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_channels: int,
                 spiral_indices: list, down_transform: list,num_classes=3) -> None:
        super(SpiralNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.num_vert = self.down_transform[-1].size(0)
        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))
        self.clsf_out = torch.nn.Linear(latent_channels, num_classes)

        self.reset_parameters()

    @property
    def input_names(self) -> Sequence[str]:
        return ("mesh",)

    @property
    def output_names(self) -> Sequence[str]:
        return (
            "logits",
        )

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def forward(self, mesh):
        z = self.encoder(mesh.x)
        out = self.clsf_out(z)
        return {"logits":out}
