import numpy as np
from scipy.optimize import linprog


def lp(A, b, tol):
    # LP Solve Basis Pursuit via linear programing
    #
    # Solves the following problem:
    #   min_x || x ||_1 s.t. b = Ax
    #
    # We convert the problem by splitting x into the
    # positive and negative entries x=u-v, u,v>=0.
    #
    # The solution is returned in the vector x.

    # Set the options to be used by the linprog solver
    opt = {"tol": tol, "disp": False}

    x_dim = A.shape[1]
    A_extended = np.c_[A, -A]

    res = linprog(
        c=np.ones(A_extended.shape[1]), A_eq=A_extended, b_eq=b, options=opt)
    x_extended = res.x
    x = x_extended[: x_dim] - x_extended[x_dim:]

    return x
