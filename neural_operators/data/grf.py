"""Gaussian random field (GRF) sampler on a periodic 2D torus.

Matches the paper's setup (Section 7.3): the initial vorticity field is drawn from
    a ~ N(0, tau^(2*alpha - d) * (-Delta + tau^2 I)^(-alpha))
with d = 2, alpha = 2.5, tau = 7 by default. Sampling is done in Fourier space: for a
Gaussian field with this covariance, the Fourier coefficients are independent complex
Gaussians whose variance is given by the operator's eigenvalues (the "sqrt_eig" below),
so we can sample by drawing white noise in Fourier space, scaling it, and taking an
inverse FFT.
"""
import math

import torch


class GaussianRF2d:
    """Sampler for a mean-zero Gaussian random field on the 2D periodic torus [0,1)^2.

    The covariance operator is C = sigma^2 * (-Delta + tau^2 I)^(-alpha), whose
    eigenvalues in the Fourier basis are
        sigma^2 * (4*pi^2*|k|^2 + tau^2)^(-alpha),
    for wavenumber k = (k_x, k_y). Larger alpha / tau produce smoother fields.
    """

    def __init__(self, size, alpha=2.5, tau=7.0, sigma=None, device=None):
        self.size = size
        self.device = device
        d = 2
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - d))

        k_max = size // 2
        wavenumbers = torch.cat(
            (torch.arange(0, k_max, device=device), torch.arange(-k_max, 0, device=device)), 0
        ).repeat(size, 1)
        k_x = wavenumbers.transpose(0, 1)
        k_y = wavenumbers

        eig = (size ** 2) * math.sqrt(2.0) * sigma * (
            (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0)
        )
        eig[0, 0] = 0.0  # zero out the mean (DC) component
        self.sqrt_eig = eig

    def sample(self, n_samples):
        """Returns n_samples fields of shape (n_samples, size, size), real-valued."""
        noise = torch.randn(n_samples, self.size, self.size, dtype=torch.cfloat, device=self.device)
        coeffs = self.sqrt_eig * noise
        field = torch.fft.ifft2(coeffs).real
        return field
