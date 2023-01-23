import torch
import torch.nn as nn

from einops import rearrange

from models.cae import ComplexLayers
from models.cae.ComplexEncoder import ComplexEncoder
from models.cae.ComplexDecoder import ComplexDecoder

class ComplexAutoEncoder(nn.Module):
    def __init__(self, in_channel, in_width, in_height, hidden_dim, linear_dim):
        super(ComplexAutoEncoder, self).__init__()

        self.collapse = ComplexLayers.ComplexCollapse()

        self.encoder = ComplexEncoder(
            in_channel, in_width, in_height, hidden_dim, linear_dim
        )

        self.decoder = ComplexDecoder(
            self.encoder.feat_dims,
            in_channel,
            in_width,
            in_height,
            hidden_dim,
            linear_dim,
        )

        self.output_model = nn.Conv2d(in_channel, in_channel, 1, 1)

        self._init_output_model()

    def _init_output_model(self):
        nn.init.constant_(self.output_model.weight, 1)
        nn.init.constant_(self.output_model.bias, 0)

    def forward(self, x):
        complex_input = self.collapse(x, torch.zeros_like(x))

        z = self.encoder(complex_input)
        complex_output = self.decoder(z)

        # Handle output.
        output_magnitude = complex_output.abs()
        reconstruction = self.output_model(output_magnitude)
        reconstruction = torch.sigmoid(reconstruction)

        return reconstruction, complex_output
