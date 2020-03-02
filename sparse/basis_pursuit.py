import numpy as np
from scipy.optimize import linprog


def basis_pursuit(mat_a, b, tol=1e-4):
    r"""
    Basis Pursuit (BP) solves

    .. math::
        \min_x \|x\|_1 \quad \text{s.t.} \ \boldsymbol{A} \vec{x} = \vec{b}
        :label: bp

    via linear programming.

    The L1-relaxed constraint should approximate the original :math:`P_0`
    problem :eq:`p0_approx`.

    `scipy.optimize.linprog` is used to solve the linear programming task.

    Parameters
    ----------
    mat_a : (N, M) np.ndarray
        A fixed weight matrix :math:`\boldsymbol{A}` in the equation
        :eq:`bp`.
    b : (N,) np.ndarray
        The right side of the equation :eq:`bp`.
    tol : float
        Tolerance, the criterion to stop iterations.

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

    mat_a = np.asarray(mat_a, dtype=np.float32)
    x_dim = mat_a.shape[1]
    mat_a_extended = np.c_[mat_a, -mat_a]
    coefficients_x = np.ones(mat_a_extended.shape[1], dtype=np.float32)

    res = linprog(c=coefficients_x, A_eq=mat_a_extended, b_eq=b, options=opt)

    x_extended = res.x
    x = x_extended[: x_dim] - x_extended[x_dim:]

    return x
