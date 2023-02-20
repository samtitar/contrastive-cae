import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange

from models.cae import ComplexLayers as ComplexLayers
from torchvision.transforms.functional import resize


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


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.conv(x)
        z = self.batchnorm(z)
        return self.relu(z)


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

        self.maxpool = ComplexLayers.ComplexUpSampling2d()
        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, out_shape):
        z = self.maxpool(x, out_shape)
        z = self.conv1(z)
        return self.conv2(z)


class ComplexConvUBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = ComplexLayers.ComplexUpSampling2d()
        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)
        self.conv3 = ComplexConv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, out_shape):
        z = self.maxpool(x, out_shape)
        z = self.conv1(z)
        z = self.conv2(z)
        return self.conv3(z)


class ConvUBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = Conv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = self.upsample(x)
        z = self.conv1(z)
        return self.conv2(z)


class ConvUBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = Conv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = Conv(out_channels, out_channels, 3, 1, 1)
        self.conv3 = Conv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = self.upsample(x)
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
        num_clusters: int = 2,
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
        self.up2 = ComplexConvUBlock2(128, 64)
        self.up1 = ComplexConvUBlock2(64, in_channels)

        self.output_model = nn.Conv2d(in_channels, in_channels, 1, 1)

        self.clustering5 = ConvUBlock3(512, 512)
        self.clustering4 = ConvUBlock3(512, 256)
        self.clustering3 = ConvUBlock3(256, 128)
        self.clustering2 = ConvUBlock2(128, 64)
        self.clustering1 = ConvUBlock2(64, num_clusters)

        self.extend_and_apply = lambda x, m: x * self.collapse(
            m.expand(x.shape), m.expand(x.shape)
        )

        def compute_feature_size():
            x = torch.zeros(1, self.in_channels, in_height, in_width)
            complex_input = self.collapse(x, torch.zeros_like(x))

            z, _, _ = self.down1(complex_input)
            z, _, _ = self.down2(z)
            z, _, _ = self.down3(z)
            z, _, _ = self.down4(z)
            z, _, _ = self.down5(z)

            return z.shape[1], z.shape[2], z.shape[3]

        fc, fh, fw = compute_feature_size()

        self.linear1 = ComplexLin(fc * fh * fw, num_features)
        self.linear2 = ComplexLin(num_features, fc * fh * fw)

        self._init_output_model()

    def _init_output_model(self):
        nn.init.constant_(self.output_model.weight, 1)
        nn.init.constant_(self.output_model.bias, 0)

    def masked_train(self):
        self.eval()
        self.clustering5.train()
        self.clustering4.train()
        self.clustering3.train()
        self.clustering2.train()
        self.clustering1.train()

    def resize(self, x):
        return x

    def forward(self, x, masks=None):
        b = x.shape[0]
        complex_input = self.collapse(x, torch.zeros_like(x))

        if masks != None:
            masks = masks.float()
            complex_input = self.extend_and_apply(complex_input, masks)

        # Encode image
        z, _, shape1 = self.down1(complex_input)
        z, _, shape2 = self.down2(z)
        z, _, shape3 = self.down3(z)
        z, _, shape4 = self.down4(z)
        z, _, shape5 = self.down5(z)

        if masks != None:
            z_masks = resize(masks, z.shape[-2:])
            z = self.extend_and_apply(z, z_masks)

        # Feature flattening
        z_shape = z.shape
        z = z.flatten(start_dim=1)
        complex_latent = self.linear1(z)
        z = self.linear2(complex_latent)
        z = z.view(z_shape)

        cz = ComplexLayers.stable_angle(z)

        # Deocde features
        z5 = self.up5(z, shape5)
        z4 = self.up4(z5, shape4)
        z3 = self.up3(z4, shape3)
        z2 = self.up2(z3, shape2)
        complex_output = self.up1(z2, shape1)

        # Create cluster masks
        cz = self.clustering5(cz)
        cz = self.clustering4(cz + (ComplexLayers.stable_angle(z5) + np.pi) / np.pi * 2)
        cz = self.clustering3(cz + (ComplexLayers.stable_angle(z4) + np.pi) / np.pi * 2)
        cz = self.clustering2(cz + (ComplexLayers.stable_angle(z3) + np.pi) / np.pi * 2)
        cz = self.clustering1(cz + (ComplexLayers.stable_angle(z2) + np.pi) / np.pi * 2)
        clusters = F.softmax(cz, dim=1)

        if masks != None:
            complex_output = self.extend_and_apply(complex_output, masks)

        # Reconstruct from magnitudes and cluster from phases
        reconstruction = self.output_model(complex_output.abs()).sigmoid()
        return complex_latent, reconstruction, complex_output, clusters

    def encode(self, x, masks=None):
        complex_input = self.collapse(x, torch.zeros_like(x))

        if masks != None:
            masks = masks.float()
            complex_input = self.extend_and_apply(complex_input, masks)

        # Encode image
        z, _, shape1 = self.down1(complex_input)
        z, _, shape2 = self.down2(z)
        z, _, shape3 = self.down3(z)
        z, _, shape4 = self.down4(z)
        z, _, shape5 = self.down5(z)

        if masks != None:
            z_masks = resize(masks, z.shape[-2:])
            z = self.extend_and_apply(z, z_masks)

        # Feature flattening
        z_shape = z.shape
        z = z.flatten(start_dim=1)
        return self.linear1(z)
