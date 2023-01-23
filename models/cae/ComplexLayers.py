import math

import torch
import torch.nn as nn


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
    return magnitude * torch.exp(phase * 1j)


def complex_tensor_to_real(complex_tensor, dim=-1):
    return torch.stack([complex_tensor.real, complex_tensor.imag], dim=dim)


def tensor_to_numpy(input_tensor):
    return input_tensor.detach().cpu().numpy()


def spherical_to_cartesian_coordinates(x):
    # Second dimension of x contains spherical coordinates: (r, phi_1, ... phi_n).
    num_dims = x.shape[1]
    out = torch.zeros_like(x)

    r = x[:, 0]
    phi = x[:, 1:]

    sin_component = 1
    for i in range(num_dims - 1):
        out[:, i] = r * torch.cos(phi[:, i]) * sin_component
        sin_component = sin_component * torch.sin(phi[:, i])

    out[:, -1] = r * sin_component
    return out


def phase_to_cartesian_coordinates(opt, phase, norm_magnitude):
    # Map phases on unit-circle and transform to cartesian coordinates.
    unit_circle_phase = torch.concat(
        (torch.ones_like(phase)[:, None], phase[:, None]), dim=1
    )

    if opt.evaluation.phase_mask_threshold != -1:
        # When magnitude is < phase_mask_threshold, use as multiplier to mask out respective phases from eval.
        unit_circle_phase = unit_circle_phase * norm_magnitude[:, None]

    return spherical_to_cartesian_coordinates(unit_circle_phase)


def clip_and_rescale(input_tensor, clip_value):
    if torch.is_tensor(input_tensor):
        clipped = torch.clamp(input_tensor, min=0, max=clip_value)
    elif isinstance(input_tensor, np.ndarray):
        clipped = np.clip(input_tensor, a_min=0, a_max=clip_value)
    else:
        raise NotImplementedError

    return clipped * (1 / clip_value)


def get_complex_number(magnitude, phase):
    return magnitude * torch.exp(phase * 1j)


def complex_tensor_to_real(complex_tensor, dim=-1):
    return torch.stack([complex_tensor.real, complex_tensor.imag], dim=dim)


def stable_angle(x: torch.tensor, eps=1e-8):
    """Function to ensure that the gradients of .angle() are well behaved."""
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return y.angle()


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
    ):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_tran = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=False,
        )

        self.kernel_size = (kernel_size, kernel_size)
        fan_in = out_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = get_conv_biases(out_channels, fan_in)

    def forward(self, x):
        return apply_layer(self.conv_tran, self.phase_bias, self.magnitude_bias, x)


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
    ):
        super(ComplexConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )

        self.kernel_size = (kernel_size, kernel_size)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.magnitude_bias, self.phase_bias = get_conv_biases(out_channels, fan_in)

    def forward(self, x):
        return apply_layer(self.conv, self.phase_bias, self.magnitude_bias, x)


class ComplexLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
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

    def forward(self, x):
        return apply_layer(self.fc, self.phase_bias, self.magnitude_bias, x)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, in_channel):
        super(ComplexBatchNorm2d, self).__init__()

        self.batchnorm = nn.BatchNorm2d(in_channel, affine=True)

    def forward(self, m, phi):
        m = self.batchnorm(m)
        return m, phi


class ComplexLayerNorm(nn.Module):
    def __init__(self, in_channel):
        super(ComplexLayerNorm, self).__init__()

        self.layernorm = nn.LayerNorm(in_channel, elementwise_affine=True)

    def forward(self, m, phi):
        m = self.layernorm(m)
        return m, phi


class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, m, phi):
        m = torch.nn.functional.relu(m)
        return m, phi


class ComplexCollapse(nn.Module):
    def __init__(self):
        super(ComplexCollapse, self).__init__()

    def forward(self, m, phi):
        return get_complex_number(m, phi)


class ComplexSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
