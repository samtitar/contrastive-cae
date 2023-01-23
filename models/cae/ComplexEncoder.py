import torch
import torch.nn as nn

from einops import rearrange
from models.cae import ComplexLayers


class ComplexEncoder(nn.Module):
    def __init__(self, in_channel, in_width, in_height, hidden_dim, linear_dim):
        super().__init__()

        self.out_channel = [
            hidden_dim,
            hidden_dim,
            2 * hidden_dim,
            2 * hidden_dim,
            2 * hidden_dim,
        ]

        self.conv_layers = ComplexLayers.ComplexSequential(
            ComplexLayers.ComplexConv2d(
                in_channel,
                self.out_channel[0],
                kernel_size=3,
                padding=1,
                stride=2,
            ),  # e.g. 32x32 => 16x16.
            ComplexLayers.ComplexBatchNorm2d(self.out_channel[0]),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
            ComplexLayers.ComplexConv2d(
                self.out_channel[0],
                self.out_channel[1],
                kernel_size=3,
                padding=1,
            ),
            ComplexLayers.ComplexBatchNorm2d(self.out_channel[1]),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
            ComplexLayers.ComplexConv2d(
                self.out_channel[1],
                self.out_channel[2],
                kernel_size=3,
                padding=1,
                stride=2,
            ),  # e.g. 16x16 => 8x8.
            ComplexLayers.ComplexBatchNorm2d(self.out_channel[2]),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
            ComplexLayers.ComplexConv2d(
                self.out_channel[2],
                self.out_channel[3],
                kernel_size=3,
                padding=1,
            ),
            ComplexLayers.ComplexBatchNorm2d(self.out_channel[3]),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
            ComplexLayers.ComplexConv2d(
                self.out_channel[3],
                self.out_channel[4],
                kernel_size=3,
                padding=1,
                stride=2,
            ),  # e.g. 8x8 => 4x4.
            ComplexLayers.ComplexBatchNorm2d(self.out_channel[4]),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
        )

        self.feat_dims = self.get_feature_dimensions(in_channel, in_width, in_height)

        self.linear_layers = ComplexLayers.ComplexSequential(
            ComplexLayers.ComplexLinear(
                2 * self.feat_dims[0] * self.feat_dims[1] * hidden_dim,
                linear_dim,
            ),
            ComplexLayers.ComplexLayerNorm(linear_dim),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
        )

    def get_feature_dimensions(self, in_channel, in_width, in_height):
        x = torch.zeros(1, in_channel, in_height, in_width)

        phase = torch.zeros_like(x)
        x = ComplexLayers.get_complex_number(x, phase)
        x = self.conv_layers(x)

        return x.shape[2], x.shape[3]

    def forward(self, x):
        z = self.conv_layers(x)
        z = rearrange(z, "b c h w -> b (c h w)")
        return self.linear_layers(z)
