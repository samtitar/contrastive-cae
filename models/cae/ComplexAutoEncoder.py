import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from einops import rearrange

from models.cae import ComplexLayers as ComplexLayers
from torchvision.transforms.functional import resize


class ComplexLin(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=None):
        super().__init__()

        self.activation = activation

        self.linear = ComplexLayers.ComplexLinear(in_channels, out_channels)
        self.layernorm = ComplexLayers.ComplexLayerNorm(out_channels)
        if activation == "relu":
            self.activation_func = ComplexLayers.ComplexReLU()
        elif activation == "sigmoid":
            self.activation_func = ComplexLayers.ComplexSigmoid()
        self.collapse = ComplexLayers.ComplexCollapse()

    def forward(self, x):
        m, p = self.linear(x)
        m, p = self.layernorm(m, p)
        if self.activation != None:
            m, p = self.activation_func(m, p)
        return self.collapse(m, p)


class Lin(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=None):
        super().__init__()

        self.activation = activation

        self.linear = nn.Linear(in_channels, out_channels)
        # self.layernorm = nn.LayerNorm(out_channels)

        if activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_func = nn.Sigmoid()

    def forward(self, x):
        y = self.linear(x)
        # y = self.layernorm(y)
        if self.activation != None:
            y = self.activation_func(y)
        return y


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
        z1 = self.conv1(x)
        z = self.conv2(z1)
        z = self.conv3(z) + z1
        return self.maxpool(z) + (tuple(z.shape[2:]),)


class ComplexConvUBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = ComplexLayers.ComplexUpSampling2d()
        # self.upsample = ComplexLayers.ComplexMaxUnpool2d()
        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, idx, out_shape):
        # z = self.upsample(x, idx, out_shape)
        z = self.upsample(x, out_shape)
        z = self.conv1(z)
        return self.conv2(z)


class ComplexConvUBlock3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = ComplexLayers.ComplexUpSampling2d()
        # self.upsample = ComplexLayers.ComplexMaxUnpool2d()
        self.conv1 = ComplexConv(in_channels, out_channels, 3, 1, 1)
        self.conv2 = ComplexConv(out_channels, out_channels, 3, 1, 1)
        self.conv3 = ComplexConv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x, idx, out_shape):
        # z = self.upsample(x, idx, out_shape)
        z = self.upsample(x, out_shape)
        z1 = self.conv1(z)
        z = self.conv2(z1)
        return self.conv3(z) + z1


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

        self.phase_collapse = ComplexLayers.ComplexPhaseCollapse()

        self.output_model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1),
        )

        self._init_output_model()

    def _init_output_model(self):
        for layer in self.output_model.children():
            if hasattr(layer, "weight"):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, -0.1)

    def forward(self, x, phase=None):
        b = x.shape[0]

        if phase == None:
            phase = torch.zeros_like(x)
        complex_input = self.collapse(x, phase)

        # Encode image
        z, idx1, shape1 = self.down1(complex_input)
        z, idx2, shape2 = self.down2(z)
        z, idx3, shape3 = self.down3(z)
        z, idx4, shape4 = self.down4(z)
        z, idx5, shape5 = self.down5(z)

        # Deocde features
        z = self.up5(z, idx5, shape5)
        z = self.up4(z, idx4, shape4)
        z = self.up3(z, idx3, shape3)
        z = self.up2(z, idx2, shape2)
        z = self.up1(z, idx1, shape1)

        complex_output = self.phase_collapse(
            z.abs() + 0.1, ComplexLayers.stable_angle(z)
        )

        # Reconstruct from magnitudes and cluster from phases
        reconstruction = self.output_model(complex_output.abs()).sigmoid()
        return (
            reconstruction,
            complex_output,
            (z.abs() + 0.1, ComplexLayers.stable_angle(z)),
        )


class SegmentationHead(nn.Module):
    def __init__(self, image_height, image_width, num_clusters, num_segments):
        super().__init__()

        self.num_clusters = num_clusters
        self.num_segments = num_segments

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)

        x = torch.zeros(1, 1, image_height, image_width)
        x = F.max_pool2d(self.conv1(x).relu(), (3, 3))
        x = F.max_pool2d(self.conv2(x).relu(), (3, 3))
        x = F.max_pool2d(self.conv3(x).relu(), (2, 2))
        x = F.max_pool2d(self.conv4(x).relu(), (2, 2))
        fw, fh = x.shape[2:]

        self.fc = nn.Linear(256 * fh * fw, num_clusters * num_segments)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x).relu(), (3, 3))
        x = F.max_pool2d(self.conv2(x).relu(), (3, 3))
        x = F.max_pool2d(self.conv3(x).relu(), (2, 2))
        x = F.max_pool2d(self.conv4(x).relu(), (2, 2))

        x = self.fc(x.flatten(start_dim=1))
        return x.view(-1, self.num_clusters, self.num_segments).softmax(dim=-1)
