"""Small MLP used to parameterize the kernel function kappa(x, y) (and, for the
richer variants (8)/(9) in the paper, kappa(x, y, a(x), a(y)) or kappa(x, y, v(x), v(y)))
that appears inside the neural operator's integral kernel operator. Shared by GNO, LNO,
and MGNO, which differ in how they use this kernel network's output (dense pairwise
matrix vs. low-rank factors vs. multi-scale hierarchy) rather than in how kappa itself
is represented."""
import torch.nn as nn


class KernelMLP(nn.Module):
    """A feed-forward network mapping edge features (e.g. concat(x, y)) to a
    dv_out x dv_in matrix, flattened to a vector of size dv_out*dv_in. This lets the
    kernel operator apply a *different, learned* linear map for every pair (x, y),
    which is what makes the integral kernel operator (7) more expressive than a single
    shared matrix W."""

    def __init__(self, in_features, dv_in, dv_out, hidden=64, n_hidden_layers=2):
        super().__init__()
        self.dv_in = dv_in
        self.dv_out = dv_out

        layers = [nn.Linear(in_features, hidden), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, dv_in * dv_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, edge_features):
        # edge_features: (E, in_features) -> (E, dv_out, dv_in)
        out = self.net(edge_features)
        return out.view(-1, self.dv_out, self.dv_in)
