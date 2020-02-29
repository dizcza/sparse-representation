import numpy as np


def oracle(CA, b, s):
    # ORACLE Implementation of the Oracle estimator
    #
    # Solves the following problem:
    #   min_x ||b - CAx||_2^2 s.t. supp{x} = s
    # where s is a vector containing the support of the true sparse vector
    #
    # The solution is returned in the vector x

    # TODO: Implement the Oracle estimator
    # Write your code here... x = ????
    x = np.zeros(CA.shape[1], dtype=np.float32)
    a_support = np.take(CA, s, axis=1)
    a_inv = np.linalg.pinv(a_support)
    x[s] = a_inv.dot(b)

    return x
