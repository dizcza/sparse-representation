r"""
Basis Pursuit (BP) solvers tackle the original :math:`P_0` problem
:eq:`p0_approx` by posing L1-relaxation on the norm of unknown :math:`\vec{x}`.

.. currentmodule:: sparse.basis_pursuit

.. autosummary::
   :toctree: toctree/basis_pursuit/

   basis_pursuit_linprog
   basis_pursuit_admm

"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import linprog

from sparse.relaxation.utils import soft_shrinkage, negligible_improvement


def basis_pursuit_linprog(A, b, tol=1e-4):
    r"""
    Basis Pursuit solver for the :math:`P_1` problem

    .. math::
        \min_x \|x\|_1 \quad \text{s.t.} \ \boldsymbol{A} \vec{x} = \vec{b}
        :label: bp

    via linear programming.

    `scipy.optimize.linprog` is used to solve the linear programming task.

    Parameters
    ----------
    A : (N, M) np.ndarray
        The input weight matrix :math:`\boldsymbol{A}`.
    b : (N,) np.ndarray
        The right side of the equation :eq:`bp`.
    tol : float
        The accuracy tolerance of linprog.

    Returns
    -------
    x : (M,) np.ndarray
        :math:`\vec{x}`, the solution to :eq:`bp`.

    """
    # We convert the problem by splitting x into the
    # positive and negative entries x=u-v, u,v>=0.
    #
    # The solution is returned in the vector x.

    # Set the options to be used by the linprog solver
    opt = {"tol": tol, "disp": False}

    A = np.asarray(A, dtype=np.float32)
    x_dim = A.shape[1]
    A_extended = np.c_[A, -A]
    coefficients_x = np.ones(A_extended.shape[1], dtype=np.float32)

    res = linprog(c=coefficients_x, A_eq=A_extended, b_eq=b, options=opt)

    x_extended = res.x
    x = x_extended[: x_dim] - x_extended[x_dim:]

    return x


def basis_pursuit_admm(A, b, lambd, tol=1e-4, max_iters=100,
                       cholesky=False):
    r"""
    Basis Pursuit solver for the :math:`Q_1^\epsilon` problem

    .. math::
        \min_x \frac{1}{2} \left|\left| \boldsymbol{A}\vec{x} - \vec{b}
        \right|\right|_2^2 + \lambda \|x\|_1

    via the alternating direction method of multipliers (ADMM).

    Parameters
    ----------
    A : (N, M) np.ndarray
        The input weight matrix :math:`\boldsymbol{A}`.
    b : (N,) np.ndarray
        The right side of the equation :math:`\boldsymbol{A}\vec{x} = \vec{b}`.
    lambd : float
        The soft-shrinkage threshold :math:`\lambda`, controls the sparsity of
        :math:`\vec{x}`.
    tol : float
        The accuracy tolerance of ADMM.
    max_iters : int
        Run for at most `max_iters` iterations.
    cholesky : bool
        Whether to use the Cholesky factorization (slow, but stable) or the
        inverse (fast, but might be unstable) of a matrix.

    Returns
    -------
    v : (M,) np.ndarray
        The solution vector :math:`\vec{x}`.
    """

    # Compute the vector of inner products between the atoms and the signal
    A_dot_b = A.T.dot(b)

    # In the x-update step of the ADMM we use the Cholesky factorization for
    # solving efficiently a given linear system Ax=b. The idea of this
    # factorization is to decompose a symmetric positive-definite matrix A
    # by A = L*L^T = L*U, where L is a lower triangular matrix and U is
    # its transpose. Given L and U, we can solve Ax = b by first solving
    # Ly = b for y by forward substitution, and then solving Ux = y
    # for x by back substitution.
    # To conclude, given A and b, where A is symmetric and positive-definite,
    # we first compute L using Matlab's command L = chol( A, 'lower' );
    # and get U by setting U = L'; Then, we obtain x via x = U \ (L \ b);
    # Note that the matrix A is fixed along the iterations of the ADMM
    # (and so as L and U). Therefore, in order to reduce computations,
    # we compute its decomposition once.

    # Compute the Cholesky factorization of M = CA'*CA + I for fast
    #  computation of the x-update. Use Matlab's chol function and produce a
    #  lower triangular matrix L, satisfying the equation M = L*L'
    M = A.T.dot(A) + np.eye(A.shape[1], dtype=np.float32)
    if cholesky:
        L = np.linalg.cholesky(M)
    else:
        M_inv = np.linalg.inv(M)

    # Initialize v
    v = np.zeros(A.shape[1], dtype=np.float32)

    # Initialize u, the dual variable of ADMM
    u = np.zeros(A.shape[1], dtype=np.float32)

    # Initialize the previous estimate of v, used for convergence test
    v_prev = v.copy()

    # main loop
    for i in range(max_iters):
        # x-update via Cholesky factorization. Solve the linear system
        # (CA'*CA + I)x = (CAtb + v - u)
        b_eff = A_dot_b + v - u
        if cholesky:
            # safe, slow
            y = solve_triangular(L, b_eff, trans=0, lower=True)
            x = solve_triangular(L, y, trans=2, lower=True)
        else:
            # unsafe, fast
            x = M_inv.dot(b_eff)

        # v-update via soft thresholding
        v = soft_shrinkage(x + u, lambd=lambd)

        # u-update according to the ADMM formula
        u = u + x - v

        # Check if converged
        if negligible_improvement(v, v_prev, tol=tol):
            break

        # Save the previous estimate in v_prev
        v_prev = v.copy()

    return v
