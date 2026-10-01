"""Multipole Graph Neural Operator (MGNO), a simplified reproduction of Section 4.3.

GNO's radius-truncated message passing captures local interactions cheaply, but a
single fixed radius cannot cheaply capture *long-range* interactions: covering the
whole domain with one radius reintroduces the O(J^2) cost truncation was meant to
avoid. The paper's multipole idea (inspired by the classical Fast Multipole Method)
resolves this by building a **hierarchy of grids** at decreasing resolution and letting
long-range interactions happen on the *coarse* levels, where there are few enough
points that dense (near-global) connectivity is cheap.

This module implements a two-level version of that idea:

  1. **Fine-to-fine** message passing: ordinary radius-truncated GNO update at full
     resolution, capturing local/near-field interactions (small radius, many points --
     stays cheap because it's truncated, exactly as in GNO).
  2. **Restrict** (fine -> coarse): fine-level features are aggregated onto a much
     coarser set of points (e.g. every 4th grid point in each direction) via a
     radius-truncated kernel -- analogous to a multigrid restriction operator.
  3. **Coarse-to-coarse**: message passing among the coarse points using a *large*
     radius that makes the coarse graph effectively fully connected. Because the coarse
     point count is small, this densely-connected computation is still cheap, and it is
     exactly what lets long-range/far-field interactions be captured without O(J^2)
     cost at the fine level -- the core multipole idea.
  4. **Prolong** (coarse -> fine): the coarse level's processed features are
     broadcast back down to the fine points (again via a radius-truncated kernel,
     analogous to multigrid prolongation/interpolation) and added to the fine-to-fine
     output.

Each of the four steps reuses the same building block as GNO: a learned kernel network
producing a per-edge matrix, applied via message passing with mean aggregation (see
`neural_operators/models/gno.py` and `neural_operators/layers/kernel_nn.py`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_operators.layers.kernel_nn import KernelMLP
from neural_operators.models.gno import build_radius_graph, make_grid_coords


def _message_pass(kernel_net, v_source, edge_index, edge_features, num_target):
    source, target = edge_index[0], edge_index[1]
    batch = v_source.shape[0]
    width = v_source.shape[-1]

    kappa = kernel_net(edge_features)               # (E, width, width)
    v_src = v_source[:, source, :]                   # (batch, E, width)
    messages = torch.einsum("eoi,bei->beo", kappa, v_src)  # (batch, E, width)

    agg = torch.zeros(batch, num_target, width, device=v_source.device, dtype=v_source.dtype)
    counts = torch.zeros(num_target, device=v_source.device, dtype=v_source.dtype)
    agg.index_add_(1, target, messages)
    counts.index_add_(0, target, torch.ones_like(target, dtype=v_source.dtype))
    agg = agg / counts.clamp(min=1.0).view(1, -1, 1)
    return agg


class MGNOLayer(nn.Module):
    def __init__(self, width, coord_dim=2, hidden=32):
        super().__init__()
        self.fine_kernel = KernelMLP(2 * coord_dim, width, width, hidden=hidden)
        self.restrict_kernel = KernelMLP(2 * coord_dim, width, width, hidden=hidden)
        self.coarse_kernel = KernelMLP(2 * coord_dim, width, width, hidden=hidden)
        self.prolong_kernel = KernelMLP(2 * coord_dim, width, width, hidden=hidden)
        self.local_linear = nn.Linear(width, width)

    def forward(self, v_fine, graph):
        fine_msg = _message_pass(
            self.fine_kernel, v_fine[:, graph["fine_key_idx"], :],
            graph["fine_edge_index"], graph["fine_edge_features"], graph["n_fine"],
        )

        v_coarse = _message_pass(
            self.restrict_kernel, v_fine,
            graph["restrict_edge_index"], graph["restrict_edge_features"], graph["n_coarse"],
        )
        v_coarse = F.gelu(v_coarse)

        coarse_msg = _message_pass(
            self.coarse_kernel, v_coarse,
            graph["coarse_edge_index"], graph["coarse_edge_features"], graph["n_coarse"],
        )

        prolonged = _message_pass(
            self.prolong_kernel, coarse_msg,
            graph["prolong_edge_index"], graph["prolong_edge_features"], graph["n_fine"],
        )

        return fine_msg + prolonged + self.local_linear(v_fine)


class MGNO2d(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        width=16,
        n_layers=4,
        fine_radius=0.08,
        restrict_radius=0.12,
        coarse_stride=4,
        n_fine_subsample=384,
        seed=0,
    ):
        super().__init__()
        self.width = width
        self.fine_radius = fine_radius
        self.restrict_radius = restrict_radius
        self.coarse_stride = coarse_stride
        self.n_fine_subsample = n_fine_subsample
        self.seed = seed

        self.lift = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([MGNOLayer(width) for _ in range(n_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)

        self._graph_cache = {}

    def _build_graph(self, size_x, size_y, device):
        fine_coords = make_grid_coords(size_x, size_y, device=device)
        n_fine = fine_coords.shape[0]

        g = torch.Generator(device="cpu").manual_seed(self.seed)
        n_sub = min(self.n_fine_subsample, n_fine)
        fine_key_idx = torch.randperm(n_fine, generator=g)[:n_sub].to(device)
        fine_key_coords = fine_coords[fine_key_idx]

        fine_edge_index = build_radius_graph(fine_coords, fine_key_coords, self.fine_radius)
        fine_edge_features = torch.cat(
            (fine_coords[fine_edge_index[1]], fine_key_coords[fine_edge_index[0]]), dim=-1
        )

        coarse_size_x = max(2, size_x // self.coarse_stride)
        coarse_size_y = max(2, size_y // self.coarse_stride)
        coarse_coords = make_grid_coords(coarse_size_x, coarse_size_y, device=device)
        n_coarse = coarse_coords.shape[0]

        # restrict: fine (source) -> coarse (target)
        restrict_edge_index = build_radius_graph(coarse_coords, fine_coords, self.restrict_radius)
        restrict_edge_features = torch.cat(
            (coarse_coords[restrict_edge_index[1]], fine_coords[restrict_edge_index[0]]), dim=-1
        )

        # coarse <-> coarse, large radius => effectively fully connected (captures long range cheaply)
        coarse_edge_index = build_radius_graph(coarse_coords, coarse_coords, radius=2.0)
        coarse_edge_features = torch.cat(
            (coarse_coords[coarse_edge_index[1]], coarse_coords[coarse_edge_index[0]]), dim=-1
        )

        # prolong: coarse (source) -> fine (target)
        prolong_edge_index = build_radius_graph(fine_coords, coarse_coords, self.restrict_radius)
        prolong_edge_features = torch.cat(
            (fine_coords[prolong_edge_index[1]], coarse_coords[prolong_edge_index[0]]), dim=-1
        )

        return {
            "fine_key_idx": fine_key_idx,
            "fine_edge_index": fine_edge_index,
            "fine_edge_features": fine_edge_features,
            "restrict_edge_index": restrict_edge_index,
            "restrict_edge_features": restrict_edge_features,
            "coarse_edge_index": coarse_edge_index,
            "coarse_edge_features": coarse_edge_features,
            "prolong_edge_index": prolong_edge_index,
            "prolong_edge_features": prolong_edge_features,
            "n_fine": n_fine,
            "n_coarse": n_coarse,
        }

    def _get_graph(self, size_x, size_y, device):
        key = (size_x, size_y, str(device))
        if key not in self._graph_cache:
            self._graph_cache[key] = self._build_graph(size_x, size_y, device)
        return self._graph_cache[key]

    def forward(self, a):
        if a.dim() == 3:
            a = a.unsqueeze(-1)
        batch, size_x, size_y = a.shape[0], a.shape[1], a.shape[2]
        device = a.device
        graph = self._get_graph(size_x, size_y, device)
        n_points = size_x * size_y

        fine_coords = make_grid_coords(size_x, size_y, device=device)
        a_flat = a.reshape(batch, n_points, -1)
        coords_b = fine_coords.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat((a_flat, coords_b), dim=-1)
        v = self.lift(x)

        for t, layer in enumerate(self.layers):
            v = layer(v, graph)
            if t < len(self.layers) - 1:
                v = F.gelu(v)

        out = F.gelu(self.project1(v))
        out = self.project2(out)
        return out.view(batch, size_x, size_y)
