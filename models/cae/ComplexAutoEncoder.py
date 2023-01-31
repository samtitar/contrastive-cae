import torch
import torch.nn as nn

from einops import rearrange

from models.cae import ComplexLayers


class ComplexLin(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.linear = ComplexLayers.ComplexLinear(in_channels, out_channels)
        self.layernorm = ComplexLayers.ComplexLayerNorm(out_channels)
        self.relu = ComplexLayers.ComplexReLU()
        self.collapse = ComplexLayers.ComplexCollapse()

    def forward(self, x):
        m, p = self.linear(x)
        m, p = self.layernorm(m, p)
        m, p = self.relu(m, p)
        return self.collapse(m, p)


class ComplexConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.conv = ComplexLayers.ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.batchnorm = ComplexLayers.ComplexBatchNorm2d(out_channels)
        self.relu = ComplexLayers.ComplexReLU()
        self.collapse = ComplexLayers.ComplexCollapse()

    def forward(self, x):
        m, p = self.conv(x)
        m, p = self.batchnorm(m, p)
        m, p = self.relu(m, p)
        return self.collapse(m, p)


class ComplexConvDBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)
        self.maxpool = ComplexLayers.ComplexMaxPool2d((2, 2))

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        return self.maxpool(z) + (tuple(z.shape[2:]),)


class ComplexConvDBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)
        self.conv3 = ComplexConv(out_channels, out_channels, 3, 1, 1)
        self.maxpool = ComplexLayers.ComplexMaxPool2d((2, 2))

    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        return self.maxpool(z) + (tuple(z.shape[2:]),)


class ComplexConvUBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = ComplexLayers.ComplexMaxUnpool2d()
        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, indices, out_shape):
        z = self.maxpool(x, indices, out_shape)
        z = self.conv1(z)
        return self.conv2(z)


class ComplexConvUBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = ComplexLayers.ComplexMaxUnpool2d()
        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)
        self.conv3 = ComplexConv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, indices, out_shape):
        z = self.maxpool(x, indices, out_shape)
        z = self.conv1(z)
        z = self.conv2(z)
        return self.conv3(z)


class ComplexAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        in_height: int = 224,
        in_width: int = 224,
        num_features: int = 1024,
    ):
        super(ComplexAutoEncoder, self).__init__()

        self.in_channels = in_channels
        self.in_height = in_height
        self.in_width = in_width

        self.collapse = ComplexLayers.ComplexCollapse()

        self.down1 = ComplexConvDBlock2(in_channels, 64)
        self.down2 = ComplexConvDBlock2(64, 128)
        self.down3 = ComplexConvDBlock3(128, 256)
        self.down4 = ComplexConvDBlock3(256, 512)
        self.down5 = ComplexConvDBlock3(512, 512)

        self.up5 = ComplexConvUBlock3(512, 512)
        self.up4 = ComplexConvUBlock3(512, 256)
        self.up3 = ComplexConvUBlock3(256, 128)
        self.up2 = ComplexConvUBlock3(128, 64)
        self.up1 = ComplexConvUBlock3(64, in_channels)

        self.output_model = nn.Conv2d(in_channels, in_channels, 1, 1)

        def compute_feature_size():
            x = torch.zeros(1, 1, in_height, in_width)
            complex_input = self.collapse(x, torch.zeros_like(x))

            z, idx1, shape1 = self.down1(complex_input)
            z, idx2, shape2 = self.down2(z)
            z, idx3, shape3 = self.down3(z)
            z, idx4, shape4 = self.down4(z)
            z, idx5, shape5 = self.down5(z)

            return z.shape[1], z.shape[2], z.shape[3]

        fc, fh, fw = compute_feature_size()

        self.linear1 = ComplexLin(fc * fh * fw, num_features * 2)
        self.linear2 = ComplexLin(num_features * 2, num_features)
        self.linear3 = ComplexLin(num_features, num_features * 2)
        self.linear4 = ComplexLin(num_features * 2, fc * fh * fw)

        self._init_output_model()

    def _init_output_model(self):
        nn.init.constant_(self.output_model.weight, 1)
        nn.init.constant_(self.output_model.bias, 0)

    def forward(self, x):
        complex_input = self.collapse(x, torch.zeros_like(x))

        z, idx1, shape1 = self.down1(complex_input)
        z, idx2, shape2 = self.down2(z)
        z, idx3, shape3 = self.down3(z)
        z, idx4, shape4 = self.down4(z)
        z, idx5, shape5 = self.down5(z)

        z_shape = z.shape
        z = z.flatten(start_dim=1)
        z = self.linear1(z)
        complex_latent = self.linear2(z)
        z = self.linear3(complex_latent)
        z = self.linear4(z)
        z = z.view(z_shape)

        z = self.up5(z, idx5, shape5)
        z = self.up4(z, idx4, shape4)
        z = self.up3(z, idx3, shape3)
        z = self.up2(z, idx2, shape2)
        complex_output = self.up1(z, idx1, shape1)

        # Handle output
        output_magnitude = complex_output.abs()
        reconstruction = self.output_model(output_magnitude)
        reconstruction = torch.sigmoid(reconstruction)

        return complex_latent, reconstruction, complex_output


# class ComplexEncoder(nn.Module):
#     def __init__(self, in_channel, in_width, in_height, hidden_dim, linear_dim):
#         super().__init__()

#         self.out_channel = [
#             hidden_dim,
#             hidden_dim,
#             2 * hidden_dim,
#             2 * hidden_dim,
#             2 * hidden_dim,
#         ]

#         self.conv_layers = ComplexLayers.ComplexSequential(
#             ComplexLayers.ComplexConv2d(
#                 in_channel,
#                 self.out_channel[0],
#                 kernel_size=3,
#                 padding=1,
#                 stride=2,
#             ),  # e.g. 32x32 => 16x16.
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[0]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConv2d(
#                 self.out_channel[0],
#                 self.out_channel[1],
#                 kernel_size=3,
#                 padding=1,
#             ),
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[1]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConv2d(
#                 self.out_channel[1],
#                 self.out_channel[2],
#                 kernel_size=3,
#                 padding=1,
#                 stride=2,
#             ),  # e.g. 16x16 => 8x8.
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[2]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConv2d(
#                 self.out_channel[2],
#                 self.out_channel[3],
#                 kernel_size=3,
#                 padding=1,
#             ),
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[3]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConv2d(
#                 self.out_channel[3],
#                 self.out_channel[4],
#                 kernel_size=3,
#                 padding=1,
#                 stride=2,
#             ),  # e.g. 8x8 => 4x4.
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[4]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#         )

#         self.feat_dims = self.get_feature_dimensions(in_channel, in_width, in_height)

#         self.linear_layers = ComplexLayers.ComplexSequential(
#             ComplexLayers.ComplexLinear(
#                 2 * self.feat_dims[0] * self.feat_dims[1] * hidden_dim,
#                 linear_dim,
#             ),
#             ComplexLayers.ComplexLayerNorm(linear_dim),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#         )

#     def get_feature_dimensions(self, in_channel, in_width, in_height):
#         x = torch.zeros(1, in_channel, in_height, in_width)

#         phase = torch.zeros_like(x)
#         x = ComplexLayers.get_complex_number(x, phase)
#         x = self.conv_layers(x)

#         return x.shape[2], x.shape[3]

#     def forward(self, x):
#         z = self.conv_layers(x)
#         z = rearrange(z, "b c h w -> b (c h w)")
#         return self.linear_layers(z)


# class ComplexDecoder(nn.Module):
#     def __init__(
#         self, feat_dims, out_channel, out_width, out_height, hidden_dim, linear_dim
#     ):
#         super().__init__()

#         self.feat_dims = feat_dims

#         # print(self.out_channel, self.out_width, self.out_height)

#         self.out_channel = [
#             2 * hidden_dim,
#             2 * hidden_dim,
#             hidden_dim,
#             hidden_dim,
#             out_channel,
#         ]

#         linear_out = 2 * feat_dims[0] * feat_dims[1] * hidden_dim

#         self.linear_layers = ComplexLayers.ComplexSequential(
#             ComplexLayers.ComplexLinear(linear_dim, linear_out),
#             ComplexLayers.ComplexLayerNorm(linear_out),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#         )

#         self.conv_layers = ComplexLayers.ComplexSequential(
#             ComplexLayers.ComplexConvTranspose2d(
#                 2 * hidden_dim,
#                 self.out_channel[0],
#                 kernel_size=3,
#                 output_padding=1,
#                 padding=1,
#                 stride=2,
#             ),  # e.g. 4x4 => 8x8.
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[0]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConv2d(
#                 self.out_channel[0],
#                 self.out_channel[1],
#                 kernel_size=3,
#                 padding=1,
#             ),
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[1]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConvTranspose2d(
#                 self.out_channel[1],
#                 self.out_channel[2],
#                 kernel_size=3,
#                 output_padding=1,
#                 padding=1,
#                 stride=2,
#             ),  # e.g. 8x8 => 16x16.
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[2]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConv2d(
#                 self.out_channel[2],
#                 self.out_channel[3],
#                 kernel_size=3,
#                 padding=1,
#             ),
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[3]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#             ComplexLayers.ComplexConvTranspose2d(
#                 self.out_channel[3],
#                 self.out_channel[4],
#                 kernel_size=3,
#                 output_padding=1,
#                 padding=1,
#                 stride=2,
#             ),  # e.g. 16x16 => 32x32.
#             ComplexLayers.ComplexBatchNorm2d(self.out_channel[4]),
#             ComplexLayers.ComplexReLU(),
#             ComplexLayers.ComplexCollapse(),
#         )

#     def forward(self, x):
#         z = self.linear_layers(x)

#         z = rearrange(
#             z,
#             "b (c h w) -> b c h w",
#             b=z.shape[0],
#             h=self.feat_dims[0],
#             w=self.feat_dims[1],
#         )

#         return self.conv_layers(z)


# class ComplexAutoEncoder(nn.Module):
#     def __init__(self, in_channel, in_width, in_height, hidden_dim, linear_dim):
#         super(ComplexAutoEncoder, self).__init__()

#         self.collapse = ComplexLayers.ComplexCollapse()

#         self.encoder = ComplexEncoder(
#             in_channel, in_width, in_height, hidden_dim, linear_dim
#         )

#         self.decoder = ComplexDecoder(
#             self.encoder.feat_dims,
#             in_channel,
#             in_width,
#             in_height,
#             hidden_dim,
#             linear_dim,
#         )

#         self.output_model = nn.Conv2d(in_channel, in_channel, 1, 1)

#         self._init_output_model()

#     def _init_output_model(self):
#         nn.init.constant_(self.output_model.weight, 1)
#         nn.init.constant_(self.output_model.bias, 0)

#     def forward(self, x):
#         complex_input = self.collapse(x, torch.zeros_like(x))

#         complex_latent = self.encoder(complex_input)
#         complex_output = self.decoder(complex_latent)

#         # Handle output.
#         output_magnitude = complex_output.abs()
#         reconstruction = self.output_model(output_magnitude)
#         reconstruction = torch.sigmoid(reconstruction)

#         return complex_latent, reconstruction, complex_output
