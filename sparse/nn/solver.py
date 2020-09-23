r"""
Basis Pursuit (BP) solvers (refer to :ref:`relaxation`) PyTorch API.

.. autosummary::
   :toctree: toctree/nn/

   basis_pursuit_admm

"""

import torch
import torch.nn.functional as F

from mighty.monitor.var_online import MeanOnline

__all__ = [
    "basis_pursuit_admm"
]


def _negligible_improvement(x, x_prev, tol: float) -> torch.BoolTensor:
    x_norm = x.norm(dim=1)
    dx_norm = (x - x_prev).norm(dim=1)
    return dx_norm / x_norm < tol


def _reduce(solved, solved_batch, x_solution, x, *args):
    if solved_batch.any():
        args_reduced = []
        unsolved_ids = torch.nonzero(~solved, as_tuple=False)
        unsolved_ids.squeeze_(dim=1)
        keys = solved_batch.nonzero(as_tuple=False)
        keys.squeeze_(dim=1)
        became_solved_ids = unsolved_ids[keys]
        x_solution[became_solved_ids] = x[keys]
        solved[became_solved_ids] = True
        mask_unsolved = ~solved_batch
        x = x[mask_unsolved]
        args_reduced.append(x)
        for arg in args:
            arg = arg[mask_unsolved]
            args_reduced.append(arg)
    else:
        args_reduced = [x, *args]
    return args_reduced


def basis_pursuit_admm(A, b, lambd, M_inv=None, tol=1e-4, max_iters=100,
                       return_stats=False):
    r"""
    Basis Pursuit solver for the :math:`Q_1^\epsilon` problem

    .. math::
        \min_x \frac{1}{2} \left|\left| \boldsymbol{A}\vec{x} - \vec{b}
        \right|\right|_2^2 + \lambda \|x\|_1

    via the alternating direction method of multipliers (ADMM).

    Parameters
    ----------
    A : (N, M) torch.Tensor
        The input weight matrix :math:`\boldsymbol{A}`.
    b : (B, N) torch.Tensor
        The right side of the equation :math:`\boldsymbol{A}\vec{x} = \vec{b}`.
    lambd : float
        :math:`\lambda`, controls the sparsity of :math:`\vec{x}`.
    tol : float
        The accuracy tolerance of ADMM.
    max_iters : int
        Run for at most `max_iters` iterations.

    Returns
    -------
    torch.Tensor
        (B, M) The solution vector batch :math:`\vec{x}`.
    """
    A_dot_b = b.matmul(A)
    if M_inv is None:
        M = A.t().matmul(A) + torch.eye(A.shape[1], device=A.device)
        M_inv = M.inverse().t()
        del M

    batch_size = b.shape[0]
    v = torch.zeros(batch_size, A.shape[1], device=A.device)
    u = torch.zeros(batch_size, A.shape[1], device=A.device)
    v_prev = v.clone()

    v_solution = v.clone()
    solved = torch.zeros(batch_size, dtype=torch.bool)

    iter_id = 0
    for iter_id in range(max_iters):
        b_eff = A_dot_b + v - u
        x = b_eff.matmul(M_inv)  # M_inv is already transposed
        # x is of shape (<=B, m_atoms)
        v = F.softshrink(x + u, lambd)
        u = u + x - v
        solved_batch = _negligible_improvement(v, v_prev, tol=tol)
        v, u, A_dot_b = _reduce(solved, solved_batch, v_solution, v, u,
                                A_dot_b)
        if v.shape[0] == 0:
            # all solved
            break
        v_prev = v.clone()

    if iter_id != max_iters - 1:
        assert solved.all()
    v_solution[~solved] = v  # dump unsolved iterations

    if return_stats:
        if v.shape[0] == 0:
            dv_norm = torch.tensor(tol)
        else:
            dv_norm = (v - v_prev).norm(dim=1) / v.norm(dim=1)
        return v_solution, dv_norm.mean(), iter_id

    return v_solution


class BasisPursuitADMM:
    def __init__(self, lambd=0.1, tol=1e-4, max_iters=100):
        self.lambd = lambd
        self.tol = tol
        self.max_iters = max_iters
        self.online = dict(dv_norm=MeanOnline(),
                           iterations=MeanOnline())

    def solve(self, A, b, M_inv=None, save_stats=True):
        v_solution, dv_norm, iteration = basis_pursuit_admm(
            A=A, b=b, lambd=self.lambd,
            M_inv=M_inv, tol=self.tol,
            max_iters=self.max_iters,
            return_stats=True)
        if save_stats:
            iteration = torch.tensor(iteration + 1, dtype=torch.float32)
            self.online['dv_norm'].update(dv_norm.cpu())
            self.online['iterations'].update(iteration)
        return v_solution

    def reset_statistics(self):
        for online in self.online.values():
            online.reset()

    def __str__(self):
        return f"{self.__class__.__name__}(lambd={self.lambd}, " \
               f"tol={self.tol}, max_iters={self.max_iters})"
