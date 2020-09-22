r"""
PyTorch implementation of AutoEncoders that form sparse representations.

.. autosummary::
   :toctree: toctree/include/

   MatchingPursuit
   LISTA

"""

import torch
import torch.nn as nn

from mighty.models.autoencoder import AutoencoderOutput
from sparse.nn.solver import basis_pursuit_admm

__all__ = [
    "MatchingPursuit",
    "Softshrink",
    "LISTA"
]


class MatchingPursuit(nn.Module):
    r"""
    Basis Matching Pursuit (ADMM) AutoEncoder neural network for sparse coding.

    Parameters
    ----------
    in_features : int
        The num. of input features (X dimension).
    out_features : int
        The dimensionality of the embedding vector Z.
    lambd : float
        Sparsity penalty for the :math:`Q_1^\epsilon` problem (see
        :func:`sparse.nn.solver.basis_pursuit_admm`).
    **solver_kwargs
        Passed to `basis_pursuit_admm` solver.

    Notes
    -----
    In overcomplete coding, where sparse representations emerge,
    :code:`out_features >> in_features`. If :code:`out_features ≲ in_features`,
    the encoding representation will be dense.

    See Also
    --------
    sparse.nn.solver.basis_pursuit_admm : Basis Matching Pursuit solver,
                                          used in this model
    """
    def __init__(self, in_features, out_features, lambd=0.2, **solver_kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lambd = lambd
        self.solver_kwargs = solver_kwargs
        weight = torch.randn(out_features, in_features)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, lambd=None):
        """
        AutoEncoder forward pass.

        Parameters
        ----------
        x : (B, C, H, W) torch.Tensor
            A batch of input images

        Returns
        -------
        z : (B, Z) torch.Tensor
            Embedding vectors: sparse representation of `x`.
        decoded : (B, C, H, W) torch.Tensor
            Reconstructed `x` from `z`.
        """
        input_shape = x.shape
        if lambd is None:
            lambd = self.lambd
        x = x.flatten(start_dim=1)
        with torch.no_grad():
            self.normalize_weight()
            z = basis_pursuit_admm(A=self.weight.t(), b=x,
                                         lambd=lambd, **self.solver_kwargs)
        decoded = z.matmul(self.weight)
        return AutoencoderOutput(z, decoded.view(*input_shape))

    def normalize_weight(self):
        w_norm = self.weight.norm(p=2, dim=1).unsqueeze(dim=1)
        self.weight.div_(w_norm)

    def extra_repr(self):
        return f"in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"lambd={self.lambd}, " \
               f"solver_kwargs={self.solver_kwargs}"


class Softshrink(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lambd = nn.Parameter(torch.rand(n_features))
        self.relu = nn.ReLU()

    def forward(self, x):
        lambd = self.relu(self.lambd)  # lambda threshold must be positive
        mask1 = x > lambd
        mask2 = x < -lambd
        out = torch.zeros_like(x)
        out += mask1.float() * -lambd + mask1.float() * x
        out += mask2.float() * lambd + mask2.float() * x
        return out

    def extra_repr(self):
        return f"[learnable] n_features={self.lambd.nelement()}"


class LISTA(nn.Module):
    r"""
    Learned Iterative Shrinkage-Thresholding Algorithm [1]_ AutoEncoder neural
    network for sparse coding.

    Parameters
    ----------
    in_features : int
        The num. of input features (X dimension).
    out_features : int
        The dimensionality of the embedding vector Z.
    n_folds : int
        The num. of recursions to apply to get better convergence of Z.
        Must be greater or equal to 1.

    Notes
    -----
    In overcomplete coding, where sparse representations emerge,
    :code:`out_features >> in_features`. If :code:`out_features ≲ in_features`,
    the encoding representation will be dense.

    References
    ----------
    1. Gregor, K., & LeCun, Y. (2010, June). Learning fast approximations of
       sparse coding. In Proceedings of the 27th international conference on
       international conference on machine learning (pp. 399-406).
    """

    def __init__(self, in_features, out_features, n_folds=2):
        super().__init__()
        assert n_folds >= 1
        self.n_folds = n_folds
        self.weight_input = nn.Parameter(
            torch.Tensor(out_features, in_features))  # W_e matrix
        self.weight_lateral = nn.Parameter(
            torch.Tensor(out_features, out_features))  # S matrix
        self.soft_shrink = Softshrink(out_features)
        self.reset_parameters()

    @property
    def lambd(self):
        r"""
        Learned Softshrink threshold vector of size :code:`out_features`.
        """
        return self.soft_shrink.lambd

    def reset_parameters(self):
        # kaiming preserves the weights variance norm, compared to randn()
        nn.init.kaiming_normal_(self.weight_input, a=1)
        nn.init.kaiming_normal_(self.weight_lateral, a=1)
        nn.init.uniform_(self.soft_shrink.lambd, a=0.1, b=0.9)

    def forward(self, x):
        """
        AutoEncoder forward pass.

        Parameters
        ----------
        x : (B, C, H, W) torch.Tensor
            A batch of input images

        Returns
        -------
        z : (B, Z) torch.Tensor
            Embedding vectors: sparse representation of `x`.
        decoded : (B, C, H, W) torch.Tensor
            Reconstructed `x` from `z`.
        """
        input_shape = x.shape
        x = x.flatten(start_dim=1)  # (B, In)
        b = x.matmul(self.weight_input.T)  # (B, Out)
        z = self.soft_shrink(b)  # (B, Out)
        for recursive_step in range(self.n_folds - 1):
            z = self.soft_shrink(b + z.matmul(self.weight_lateral.T))

        decoded = z.matmul(self.weight_input)  # (B, In)
        return AutoencoderOutput(z, decoded.view(*input_shape))
