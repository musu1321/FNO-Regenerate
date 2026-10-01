"""Low-rank Neural Operator (LNO), following Section 4.2 of the paper.

Instead of truncating the integral (GNO) or restricting to translation-invariant
kernels (FNO), LNO restricts the kernel itself to be **low rank**:
    kappa(x, y) ~= sum_{r=1}^{R} phi_r(x) psi_r(y)^T
where phi_r(x) in R^{dv_out} and psi_r(y) in R^{dv_in} are learned basis functions
(parameterized by small coordinate -> R*d MLPs). Substituting into the integral
operator lets us swap the order of integration and summation:
    (K v)(x) = integral kappa(x, y) v(y) dy
             = sum_r phi_r(x) * [ integral psi_r(y)^T v(y) dy ]
             = sum_r phi_r(x) * s_r,          s_r := integral psi_r(y) . v(y) dy
The bracketed term s_r is a single scalar per basis function r, computed *once* by a
sum over all J grid points (cost O(J)), and then broadcast back out to every query
point x via phi_r(x) (also cost O(J) total, since there are J query points). So the
whole operator costs O(R * J) instead of the naive O(J^2) -- as long as we keep the
rank R small, this is very cheap, at the price of expressiveness (a rank-R kernel can
only represent kernels well-approximated by R "modes", similar in spirit to a truncated
SVD / low-rank matrix factorization).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisMLP(nn.Module):
    """Maps a coordinate x in R^{coord_dim} to R basis vectors, each of dimension d,
    i.e. output shape (N, R, d)."""

    def __init__(self, coord_dim, rank, d, hidden=64):
        super().__init__()
        self.rank = rank
        self.d = d
        self.net = nn.Sequential(
            nn.Linear(coord_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, rank * d),
        )

    def forward(self, coords):
        out = self.net(coords)  # (N, rank*d)
        return out.view(-1, self.rank, self.d)


class LNOLayer(nn.Module):
    def __init__(self, width, rank=16, coord_dim=2, hidden=64):
        super().__init__()
        self.phi_net = BasisMLP(coord_dim, rank, width, hidden=hidden)  # phi_r(x), output-side basis
        self.psi_net = BasisMLP(coord_dim, rank, width, hidden=hidden)  # psi_r(y), input-side basis
        self.local_linear = nn.Linear(width, width)  # W_t, the local term in Eq. (10)

    def forward(self, v, coords):
        # v: (batch, N, width), coords: (N, coord_dim)
        n_points = coords.shape[0]
        phi = self.phi_net(coords)  # (N, R, width)
        psi = self.psi_net(coords)  # (N, R, width)

        # s_r = (1/N) * sum_y psi_r(y) . v(y)   -- a Monte-Carlo estimate of the integral.
        s = torch.einsum("nrd,bnd->br", psi, v) / n_points  # (batch, R)

        # u(x) = sum_r phi_r(x) * s_r
        integral_term = torch.einsum("nrd,br->bnd", phi, s)  # (batch, N, width)

        return integral_term + self.local_linear(v)


class LNO2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, width=32, n_layers=4, rank=16):
        super().__init__()
        self.width = width
        self.lift = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([LNOLayer(width, rank=rank) for _ in range(n_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)
        self._coord_cache = {}

    def _get_coords(self, size_x, size_y, device):
        key = (size_x, size_y, str(device))
        if key not in self._coord_cache:
            gx = torch.linspace(0, 1, size_x, device=device)
            gy = torch.linspace(0, 1, size_y, device=device)
            xx, yy = torch.meshgrid(gx, gy, indexing="ij")
            self._coord_cache[key] = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        return self._coord_cache[key]

    def forward(self, a):
        if a.dim() == 3:
            a = a.unsqueeze(-1)
        batch, size_x, size_y = a.shape[0], a.shape[1], a.shape[2]
        device = a.device
        coords = self._get_coords(size_x, size_y, device)
        n_points = size_x * size_y

        a_flat = a.reshape(batch, n_points, -1)
        coords_b = coords.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat((a_flat, coords_b), dim=-1)
        v = self.lift(x)

        for t, layer in enumerate(self.layers):
            v = layer(v, coords)
            if t < len(self.layers) - 1:
                v = F.gelu(v)

        out = F.gelu(self.project1(v))
        out = self.project2(out)
        return out.view(batch, size_x, size_y)
