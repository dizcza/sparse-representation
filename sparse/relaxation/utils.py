import numpy as np


def soft_shrinkage(x, lambd):
    r"""
    Applies the soft shrinkage function elementwise:

    .. math::
        h_\lambda(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Parameters
    ----------
    x : np.ndarray
        Input vector.
    lambd : float
        Soft shrinkage threshold value.

    Returns
    -------
    x_soft : np.ndarray
        :math:`h_\lambda(x)`
    """
    x_soft = np.zeros_like(x)  # 0 if |x| < lmbda
    mask_less = x <= -lambd
    mask_greater = x >= lambd
    x_soft[mask_less] = x[mask_less] + lambd
    x_soft[mask_greater] = x[mask_greater] - lambd
    return x_soft


def negligible_improvement(x, x_prev, tol):
    x_norm = np.linalg.norm(x)
    dx_norm = np.linalg.norm(x - x_prev)
    return x_norm > 0 and dx_norm / x_norm < tol


def negligible_improvement_axis(x, x_prev, tol):
    x_norm = x.norm(dim=1)
    dx_norm = (x - x_prev).norm(dim=1)
    return dx_norm / x_norm < tol
