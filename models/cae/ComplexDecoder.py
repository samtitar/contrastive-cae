import torch.nn as nn

from einops import rearrange
from models.cae import ComplexLayers


class ComplexDecoder(nn.Module):
    def __init__(
        self, feat_dims, out_channel, out_width, out_height, hidden_dim, linear_dim
    ):
        super().__init__()

        self.feat_dims = feat_dims

        # print(self.out_channel, self.out_width, self.out_height)

        self.out_channel = [
            2 * hidden_dim,
            2 * hidden_dim,
            hidden_dim,
            hidden_dim,
            out_channel,
        ]

        linear_out = 2 * feat_dims[0] * feat_dims[1] * hidden_dim

        self.linear_layers = ComplexLayers.ComplexSequential(
            ComplexLayers.ComplexLinear(linear_dim, linear_out),
            ComplexLayers.ComplexLayerNorm(linear_out),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
        )

        self.conv_layers = ComplexLayers.ComplexSequential(
            ComplexLayers.ComplexConvTranspose2d(
                2 * hidden_dim,
                self.out_channel[0],
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # e.g. 4x4 => 8x8.
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
            ComplexLayers.ComplexConvTranspose2d(
                self.out_channel[1],
                self.out_channel[2],
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # e.g. 8x8 => 16x16.
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
            ComplexLayers.ComplexConvTranspose2d(
                self.out_channel[3],
                self.out_channel[4],
                kernel_size=3,
                output_padding=1,
                padding=1,
                stride=2,
            ),  # e.g. 16x16 => 32x32.
            ComplexLayers.ComplexBatchNorm2d(self.out_channel[4]),
            ComplexLayers.ComplexReLU(),
            ComplexLayers.ComplexCollapse(),
        )

    def forward(self, x):
        z = self.linear_layers(x)

        z = rearrange(
            z,
            "b (c h w) -> b c h w",
            b=z.shape[0],
            h=self.feat_dims[0],
            w=self.feat_dims[1],
        )

        return self.conv_layers(z)
