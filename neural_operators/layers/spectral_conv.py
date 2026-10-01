"""SpectralConv2d: the FNO's parameterization of the integral kernel operator (Eq. 7
in the paper) via multiplication in the Fourier domain.

Why Fourier multiplication implements a global integral operator:
For a *translation-invariant* kernel, kappa(x, y) = kappa(x - y), the integral
operator (13) becomes a convolution:
    (K v)(x) = integral kappa(x - y) v(y) dy = (kappa * v)(x)
By the convolution theorem, this is diagonal in the Fourier basis:
    F[K v] = F[kappa] . F[v]      (elementwise product of Fourier coefficients)
So instead of learning kappa(x, y) directly (which costs O(J^2) to apply, J = number
of grid points), FNO directly parameterizes a *finite* set of Fourier coefficients
R_k = F[kappa](k) for the lowest k_max modes (truncating high frequencies acts as an
implicit smoothing prior, and keeps the parameter count independent of the grid
resolution J -- this is exactly what makes FNO discretization-invariant). Applying the
operator is then: FFT -> multiply by learned complex weights R_k for |k| <= k_max
(zero out the rest) -> inverse FFT. Cost: O(J log J) instead of O(J^2).
"""
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of retained Fourier modes, first spatial dim
        self.modes2 = modes2  # number of retained Fourier modes, second spatial dim

        scale = 1.0 / (in_channels * out_channels)
        # Two separate weight blocks are needed because rfft2's output only stores the
        # non-negative frequencies along the last axis; the corresponding negative
        # frequencies along the *other* axis still need their own independent weights
        # (real signals have conjugate-symmetric spectra, but that symmetry links a
        # positive-k1 mode to a negative-k1 mode, not to itself).
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mul2d(x, weight):
        # x:      (batch, in_channels,  modes1, modes2)
        # weight: (in_channels, out_channels, modes1, modes2)
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x):
        # x: (batch, in_channels, size1, size2)
        batch_size = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]

        x_ft = torch.fft.rfft2(x)  # (batch, in_channels, size1, size2//2 + 1), complex

        out_ft = torch.zeros(
            batch_size, self.out_channels, size1, size2 // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        # Low positive frequencies (top-left block of the truncated spectrum).
        out_ft[:, :, : self.modes1, : self.modes2] = self._complex_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weight1
        )
        # Low negative frequencies along dim -2 (wrap-around; bottom-left block).
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._complex_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weight2
        )

        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        return x
