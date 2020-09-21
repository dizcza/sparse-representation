r"""
Iterative Shrinkage Algorithm (ISTA) is also used to find an approximate
solution to :math:`\text{P}_0` problem, but Basis Pursuit methods are superior.

.. autosummary::
   :toctree: toctree/relaxation/

   ista

"""

import numpy as np

from sparse.relaxation.utils import soft_shrinkage, negligible_improvement


def ista(A, b, alpha, tol=1e-4, max_iters=100):
    r"""
    Iterative Shrinkage Algorithm (ISTA) for the :math:`Q_1^\epsilon` problem:

    .. math::
        \min_x \frac{1}{2} \left|\left| \boldsymbol{A}\vec{x} - \vec{b}
        \right|\right|_2^2 + \lambda \|x\|_1

    Parameters
    ----------
    A : (N, M) np.ndarray
        The input weight matrix :math:`\boldsymbol{A}`.
    b : (N,) np.ndarray
        The right side of the equation :math:`\boldsymbol{A}\vec{x} = \vec{b}`.
    lambd : float
        :math:`\lambda`, controls the sparsity of :math:`\vec{x}`.
    tol : float
        The accuracy tolerance of ISTA.
    max_iters : int
        Run for at most `max_iters` iterations.

    Returns
    -------
    x : (M,) np.ndarray
        The solution vector batch :math:`\vec{x}`.
    """
    eigvals = np.linalg.eigvals(A.T.dot(A))

    # 1) abs() is technically not needed since the eigenvalues of a positive
    #    semi-definite symmetric matrix are non-negative
    # 2) multiplied by '2' because we need L > the-largest-eigval
    eigval_largest = 2 * np.max(np.abs(eigvals))

    alpha_norm = alpha / eigval_largest
    m_atoms = A.shape[1]
    x = np.zeros(m_atoms, dtype=np.float32)
    x_prev = x.copy()
    for iter_id in range(max_iters):
        # x_unconstrained is before applying L1 norm constraint
        x_unconstrained = x - A.T.dot(A.dot(x) - b) / eigval_largest
        x = soft_shrinkage(x_unconstrained, lambd=alpha_norm)
        if negligible_improvement(x, x_prev, tol=tol):
            break
        x_prev = x.copy()
    return x
