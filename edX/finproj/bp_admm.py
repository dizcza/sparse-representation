import numpy as np
from scipy.linalg import solve_triangular


def soft_threshold(x, lmbda):
    x_soft = np.zeros_like(x)  # 0 if |x| < lmbda
    mask_less = x <= -lmbda
    mask_greater = x >= lmbda
    x_soft[mask_less] = x[mask_less] + lmbda
    x_soft[mask_greater] = x[mask_greater] - lmbda
    return x_soft


def bp_admm(CA, b, lmbda, cholesky=False):
    # BP_ADMM Solve Basis Pursuit problem via ADMM
    #
    # Solves the following problem:
    #   min_x 1/2*||b - CAx||_2^2 + lambda*|| x ||_1
    #
    # The solution is returned in the vector v.

    # Set the accuracy tolerance of ADMM, run for at most max_admm_iters
    tol_admm = 1e-4
    max_admm_iters = 100

    # TODO: Compute the vector of inner products between the atoms and the
    #  signal
    CA_transposed_b = CA.T.dot(b)

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

    # TODO: Compute the Cholesky factorization of M = CA'*CA + I for fast
    #  computation of the x-update. Use Matlab's chol function and produce a
    #  lower triangular matrix L, satisfying the equation M = L*L'
    M = CA.T.dot(CA) + np.eye(CA.shape[1], dtype=np.float32)
    if cholesky:
        L = np.linalg.cholesky(M)
    else:
        M_inv = np.linalg.inv(M)

    # Force Matlab to recognize the upper / lower triangular structure
    # L = sparse(L)
    # U = sparse(L.T)

    # TODO: Initialize v
    v = np.zeros(CA.shape[1], dtype=np.float32)

    # TODO: Initialize u, the dual variable of ADMM
    u = np.zeros(CA.shape[1], dtype=np.float32)

    # TODO: Initialize the previous estimate of v, used for convergence test
    v_prev = v.copy()

    # main loop
    for i in range(max_admm_iters):
        # TODO: x-update via Cholesky factorization. Solve the linear system
        # (CA'*CA + I)x = (CAtb + v - u)
        # Write your code here... x = ????
        b_eff = CA_transposed_b + v - u
        if cholesky:
            y = solve_triangular(L, b_eff, trans=0, lower=True)
            x = solve_triangular(L, y, trans=2, lower=True)
        else:
            x = M_inv.dot(b_eff)

        # TODO: v-update via soft thresholding
        v = soft_threshold(x + u, lmbda=lmbda)

        # TODO: u-update according to the ADMM formula
        u = u + x - v

        # Check if converged
        if np.linalg.norm(v) > 0:
            if (np.linalg.norm(v - v_prev) / np.linalg.norm(v)) < tol_admm:
                break

        # TODO: Save the previous estimate in v_prev
        v_prev = v.copy()

    return v
