import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_layer(real_function, phase_bias, magnitude_bias, x):
    psi = real_function(x.real) + 1j * real_function(x.imag)
    m_psi = psi.abs() + magnitude_bias
    phi_psi = stable_angle(psi) + phase_bias

    chi = real_function(x.abs()) + magnitude_bias
    m = 0.5 * m_psi + 0.5 * chi

    return m, phi_psi


def get_conv_biases(out_channels, fan_in):
    magnitude_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
    magnitude_bias = init_magnitude_bias(fan_in, magnitude_bias)

    phase_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
    phase_bias = init_phase_bias(phase_bias)
    return magnitude_bias, phase_bias


def init_phase_bias(bias):
    return nn.init.constant_(bias, val=0)


def init_magnitude_bias(fan_in, bias):
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)
    return bias


def get_complex_number(magnitude, phase):
    return magnitude * torch.exp(1j * phase)


def complex_tensor_to_real(complex_tensor, dim=-1):
    return torch.stack([complex_tensor.real, complex_tensor.imag], dim=dim)


def stable_angle(x, eps=1e-8):
    """Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return y.angle()


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )

        self.kernel_size = (kernel_size, kernel_size)
        fan_in = out_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = get_conv_biases(out_channels, fan_in)

    def forward(self, x: torch.cfloat):
        return apply_layer(self.conv_tran, self.phase_bias, self.magnitude_bias, x)


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super(ComplexConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.kernel_size = (kernel_size, kernel_size)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = get_conv_biases(out_channels, fan_in)

    def forward(self, x: torch.cfloat):
        return apply_layer(self.conv, self.phase_bias, self.magnitude_bias, x)


class ComplexLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ComplexLinear, self).__init__()

        self.fc = nn.Linear(in_channels, out_channels, bias=False)

        self.magnitude_bias, self.phase_bias = self._get_biases(
            in_channels, out_channels
        )

    def _get_biases(self, in_channels, out_channels):
        fan_in = in_channels
        magnitude_bias = nn.Parameter(torch.empty((1, out_channels)))
        magnitude_bias = init_magnitude_bias(fan_in, magnitude_bias)

        phase_bias = nn.Parameter(torch.empty((1, out_channels)))
        phase_bias = init_phase_bias(phase_bias)
        return magnitude_bias, phase_bias

    def forward(self, x: torch.cfloat):
        return apply_layer(self.fc, self.phase_bias, self.magnitude_bias, x)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, in_channels: int):
        super(ComplexBatchNorm2d, self).__init__()

        self.batchnorm = nn.BatchNorm2d(in_channels, affine=True)

    def forward(self, m: torch.FloatTensor, phi: torch.FloatTensor):
        return self.batchnorm(m), phi


class ComplexLayerNorm(nn.Module):
    def __init__(self, in_channels: int):
        super(ComplexLayerNorm, self).__init__()

        self.layernorm = nn.LayerNorm(in_channels, elementwise_affine=True)

    def forward(self, m: torch.FloatTensor, phi: torch.FloatTensor):
        m = self.layernorm(m)
        return m, phi


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, m: torch.FloatTensor, phi: torch.FloatTensor):
        m = torch.nn.functional.relu(m)
        return m, phi


class ComplexSigmoid(nn.Module):
    def __init__(self):
        super(ComplexSigmoid, self).__init__()

    def forward(self, m: torch.FloatTensor, phi: torch.FloatTensor):
        m = torch.sigmoid(m)
        return m, phi


class ComplexCollapse(nn.Module):
    def __init__(self):
        super(ComplexCollapse, self).__init__()

    def forward(self, m: torch.FloatTensor, phi: torch.FloatTensor):
        return get_complex_number(m, phi)


class ComplexPhaseCollapse(nn.Module):
    def __init__(self):
        super(ComplexPhaseCollapse, self).__init__()

    def forward(self, m: torch.FloatTensor, phi: torch.FloatTensor):
        b, c, h, w = phi.shape

        phi = stable_angle(get_complex_number(m, phi).mean(dim=1))

        if len(phi.shape) == 3:
            phi = phi.unsqueeze(1)

        return get_complex_number(m, phi.repeat(1, c, 1, 1))


class ComplexSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class ComplexMaxPool2d(nn.Module):
    def __init__(
        self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1
    ):
        super(ComplexMaxPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.cfloat):
        _, index = F.max_pool2d(
            x.abs(),
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            return_indices=True,
        )

        index = index.long()

        return (
            x.flatten(start_dim=2)
            .gather(dim=2, index=index.flatten(start_dim=2))
            .view_as(index),
            index,
        )


class ComplexAveragePool(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ComplexMaxPool2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.cfloat):
        return get_complex_number(
            F.avg_pool2d(x.abs(), self.kernel_size, self.stride, self.padding),
            F.avg_pool2d(stable_angle(x), self.kernel_size, self.stride, self.padding),
        )


class ComplexMaxUnpool2d(nn.Module):
    def __init__(self):
        super(ComplexMaxUnpool2d, self).__init__()

    def forward(self, x: torch.cfloat, index: torch.LongTensor, out_shape: tuple):
        b, c = x.shape[:2]

        m = x.abs().flatten(start_dim=2)
        index = index.flatten(start_dim=2)

        empty = torch.zeros((b, c, out_shape[0] * out_shape[1])).to(x.device).float()

        return get_complex_number(
            empty.scatter_(dim=2, index=index, src=m).view(
                b, c, out_shape[0], out_shape[1]
            ),
            F.interpolate(stable_angle(x), out_shape, mode="nearest"),
        )


class ComplexUpSampling2d(nn.Module):
    def __init__(self, mode="nearest"):
        super(ComplexUpSampling2d, self).__init__()
        self.mode = mode

    def forward(self, x: torch.cfloat, out_shape: tuple):
        return get_complex_number(
            F.interpolate(x.abs(), out_shape, mode=self.mode),
            F.interpolate(stable_angle(x), out_shape, mode=self.mode),
        )
