"""FNO2d: the Fourier Neural Operator, following Eq. (6) of the paper with the
integral kernel operator instantiated as SpectralConv2d.

    G_theta = Q o sigma_T(W_{T-1} + K_{T-1} + b_{T-1}) o ... o sigma_1(W_0+K_0+b_0) o P

- P: pointwise lifting (a(x) -> higher-dim hidden representation v_0(x)).
- Each layer: v_{t+1}(x) = sigma( W_t v_t(x) + (K_t v_t)(x) + b_t ), where K_t is a
  SpectralConv2d (the non-local integral operator) and W_t is a pointwise (1x1 conv)
  local linear operator, added exactly as prescribed in Eq. (10)/(12) -- residual-like
  local term alongside the non-local spectral term.
- Q: pointwise projection (final hidden representation -> output function u(x)).

Because P, Q, W_t, b_t act pointwise and K_t depends only on a fixed, finite set of
Fourier modes, the whole architecture is discretization-invariant: the same weights can
be applied to input functions given on any grid resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_operators.layers.spectral_conv import SpectralConv2d


class FNOBlock2d(nn.Module):
    def __init__(self, width, modes1, modes2, activation=True):
        super().__init__()
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.local_linear = nn.Conv2d(width, width, kernel_size=1)  # W_t (pointwise)
        self.activation = activation

    def forward(self, x):
        x = self.spectral_conv(x) + self.local_linear(x)
        if self.activation:
            x = F.gelu(x)
        return x


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels=3,   # e.g. (a(x), x_coord, y_coord) after the coordinate-preprocessing trick
        out_channels=1,
        width=32,
        modes1=12,
        modes2=12,
        n_layers=4,
    ):
        super().__init__()
        self.width = width

        self.lift = nn.Linear(in_channels, width)  # P: pointwise lift, applied per-pixel
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(width, modes1, modes2, activation=(t < n_layers - 1))
                for t in range(n_layers)
            ]
        )
        self.project1 = nn.Linear(width, 128)  # Q: pointwise projection (two-layer MLP)
        self.project2 = nn.Linear(128, out_channels)

    @staticmethod
    def get_grid(shape, device):
        batch, size_x, size_y = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1).repeat(batch, 1, size_y, 1)
        gy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1).repeat(batch, size_x, 1, 1)
        return torch.cat((gx, gy), dim=-1)  # (batch, size_x, size_y, 2)

    def forward(self, a):
        # a: (batch, size_x, size_y) or (batch, size_x, size_y, in_channels-2) raw input function.
        if a.dim() == 3:
            a = a.unsqueeze(-1)
        grid = self.get_grid(a.shape, a.device)
        x = torch.cat((a, grid), dim=-1)  # append (x, y) coordinates -- see "Preprocessing" in the paper

        x = self.lift(x)               # (batch, size_x, size_y, width)
        x = x.permute(0, 3, 1, 2)      # -> (batch, width, size_x, size_y) for SpectralConv2d/Conv2d

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 2, 3, 1)      # -> (batch, size_x, size_y, width)
        x = F.gelu(self.project1(x))
        x = self.project2(x)
        return x.squeeze(-1)           # (batch, size_x, size_y)
