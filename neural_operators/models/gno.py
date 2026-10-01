"""Graph Neural Operator (GNO), following Section 4.1 of the paper.

GNO approximates the integral kernel operator
    (K v)(x) = integral_D kappa(x, y) v(y) dy
by combining two ideas:

1. **Truncation**: only integrate kappa(x, y) v(y) over a local neighborhood
   s(x) = B(x, r) (points within radius r of x), turning the dense integral into a
   sparse sum -- this is the paper's Eq. (16). It also builds in a local-interaction
   inductive bias, similar to a CNN's receptive field.
2. **Nystrom approximation**: instead of summing over *all* points in the
   discretization, sum only over a fixed, randomly-subsampled subset of J' << J
   "key" points -- a Monte-Carlo / low-rank approximation of the kernel matrix (Eq. 15).

Together, these reduce the O(J^2) cost of the naive integral to roughly O(J * J'),
where J' is the (small, fixed) size of the key-point subsample.

Once the graph (which query points are connected to which key points) is fixed, a GNO
layer is exactly the message-passing update below the paper's Eq. (16):
    u(x_j) = (1/|N(x_j)|) * sum_{y in N(x_j)} kappa(x_j, y) v(y)
implemented with a learned kernel network kappa (see layers/kernel_nn.py) producing a
different dv_out x dv_in matrix per edge, and averaging ("mean aggregation") over each
node's neighborhood.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_operators.layers.kernel_nn import KernelMLP


def make_grid_coords(size_x, size_y, device=None):
    gx = torch.linspace(0, 1, size_x, device=device)
    gy = torch.linspace(0, 1, size_y, device=device)
    xx, yy = torch.meshgrid(gx, gy, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)  # (N, 2)


def build_radius_graph(query_coords, key_coords, radius):
    """Returns edge_index (2, E) with row 0 = key (source) index into key_coords,
    row 1 = query (target) index into query_coords, for all pairs within `radius`."""
    dist = torch.cdist(query_coords, key_coords)  # (n_query, n_key)
    mask = dist <= radius
    target, source = mask.nonzero(as_tuple=True)
    edge_index = torch.stack((source, target), dim=0)
    return edge_index


class GNOLayer(nn.Module):
    def __init__(self, width, coord_dim=2, kernel_hidden=64):
        super().__init__()
        self.kernel_net = KernelMLP(2 * coord_dim, dv_in=width, dv_out=width, hidden=kernel_hidden)
        self.local_linear = nn.Linear(width, width)  # W_t, the local (pointwise) term in Eq. (10)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, v, edge_index, edge_features, num_query):
        """v: (batch, n_key, width) node features at the key points.
        edge_index: (2, E) = (source in key set, target in query set).
        edge_features: (E, 2*coord_dim) = concat(x_target, y_source) coordinates.
        num_query: number of query points (target index range).
        """
        source, target = edge_index[0], edge_index[1]
        batch = v.shape[0]

        kappa = self.kernel_net(edge_features)          # (E, width, width)
        v_src = v[:, source, :]                          # (batch, E, width)
        messages = torch.einsum("eoi,bei->beo", kappa, v_src)  # (batch, E, width)

        agg = torch.zeros(batch, num_query, v.shape[-1], device=v.device, dtype=v.dtype)
        counts = torch.zeros(num_query, device=v.device, dtype=v.dtype)
        agg.index_add_(1, target, messages)
        counts.index_add_(0, target, torch.ones_like(target, dtype=v.dtype))
        counts = counts.clamp(min=1.0).view(1, -1, 1)
        agg = agg / counts  # mean aggregation, matches Eq. after (16)

        # local linear term W_t applied to v at the *query* points -- if key set ==
        # query set (as in our stacked-layer setup) this is just self.local_linear(v).
        return agg, kappa  # kappa returned only for introspection/teaching purposes


class GNO2d(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        width=24,
        n_layers=4,
        radius=0.08,
        n_subsample=512,
        seed=0,
    ):
        super().__init__()
        self.width = width
        self.radius = radius
        self.n_subsample = n_subsample
        self.seed = seed

        self.lift = nn.Linear(in_channels, width)
        self.gno_layers = nn.ModuleList([GNOLayer(width) for _ in range(n_layers)])
        self.local_linears = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.project1 = nn.Linear(width, 128)
        self.project2 = nn.Linear(128, out_channels)

        self._graph_cache = {}

    def _get_graph(self, size_x, size_y, device):
        key = (size_x, size_y, str(device))
        if key not in self._graph_cache:
            coords = make_grid_coords(size_x, size_y, device=device)
            n_points = coords.shape[0]
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            n_sub = min(self.n_subsample, n_points)
            key_idx = torch.randperm(n_points, generator=g)[:n_sub].to(device)
            key_coords = coords[key_idx]

            edge_index = build_radius_graph(coords, key_coords, self.radius)
            edge_features = torch.cat(
                (coords[edge_index[1]], key_coords[edge_index[0]]), dim=-1
            )  # concat(x_target, y_source)
            self._graph_cache[key] = (coords, key_idx, edge_index, edge_features)
        return self._graph_cache[key]

    def forward(self, a):
        if a.dim() == 3:
            a = a.unsqueeze(-1)
        batch, size_x, size_y = a.shape[0], a.shape[1], a.shape[2]
        device = a.device

        coords, key_idx, edge_index, edge_features = self._get_graph(size_x, size_y, device)
        n_points = size_x * size_y

        a_flat = a.reshape(batch, n_points, -1)
        coords_b = coords.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat((a_flat, coords_b), dim=-1)
        v = self.lift(x)  # (batch, n_points, width)

        for t, (gno_layer, local_linear) in enumerate(zip(self.gno_layers, self.local_linears)):
            v_key = v[:, key_idx, :]
            agg, _ = gno_layer(v_key, edge_index, edge_features, num_query=n_points)
            v = agg + local_linear(v)
            if t < len(self.gno_layers) - 1:
                v = F.gelu(v)

        out = F.gelu(self.project1(v))
        out = self.project2(out)
        return out.view(batch, size_x, size_y)
