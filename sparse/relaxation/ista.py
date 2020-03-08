import numpy as np

from sparse.relaxation.utils import soft_shrinkage


def ista(A, b, alpha, tol=1e-3, iters_max=100):
    eigvals = np.linalg.eigvals(A.T.dot(A))
    eigval_largest = np.max(np.abs(eigvals))
    alpha_norm = alpha / eigval_largest
    m_atoms = A.shape[1]
    x = np.zeros(m_atoms, dtype=np.float32)
    x_prev = x.copy()
    for iter_id in range(iters_max):
        # x_unconstrained is before applying L1 norm constraint
        x_unconstrained = x - A.T.dot(A.dot(x) - b) / eigval_largest
        x = soft_shrinkage(x_unconstrained, threshold=alpha_norm)
        dx = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        if dx < tol:
            break
        x_prev = x.copy()
    return x
