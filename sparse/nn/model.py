import torch
import torch.nn as nn

from sparse.nn.solver import basis_pursuit_admm

__all__ = [
    "MatchingPursuit",
    "LISTA"
]


class MatchingPursuit(nn.Module):
    def __init__(self, in_features, out_features, lamb=0.2,
                 solver=basis_pursuit_admm):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lambd = lamb
        self.solver = solver
        weight = torch.randn(out_features, in_features)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, lambd=None):
        input_shape = x.shape
        if lambd is None:
            lambd = self.lambd
        x = x.flatten(start_dim=1)
        with torch.no_grad():
            self.normalize_weight()
            encoded = self.solver(A=self.weight.t(), b=x,
                                  lambd=lambd, max_iters=100)
        decoded = encoded.matmul(self.weight)
        return encoded, decoded.view(*input_shape)

    def normalize_weight(self):
        w_norm = self.weight.norm(p=2, dim=1).unsqueeze(dim=1)
        self.weight.div_(w_norm)

    def extra_repr(self):
        return f"in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"lambd={self.lambd}, " \
               f"solver={self.solver.__name__}"


class LISTA(nn.Module):
    # TODO: not implemented. SoftShrink lambd cannot be a parameter.

    def __init__(self, in_features, out_features, n_folds=2):
        super().__init__()
        self.n_folds = n_folds
        self.weight_x = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_inhib = nn.Parameter(
            torch.Tensor(out_features, out_features))
        shrinkage_thr = nn.Parameter(torch.Tensor(out_features))
        self.soft_shrink = nn.Softshrink(shrinkage_thr)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming preserves the weights variance norm, compared to randn()
        nn.init.kaiming_normal_(self.weight_x, a=1)
        nn.init.kaiming_normal_(self.weight_inhib, a=1)
        nn.init.uniform_(self.soft_shrink.lambd, a=0.1, b=0.9)

    def forward(self, x):
        b = self.weight_inhib.dot(x)
        z = self.soft_shrink(b)
        for recursive_step in range(self.n_folds - 1):
            z = self.soft_shrink(b + self.weight_inhib.dot(z))
        return z
