from sparse.greedy_pursuit import orthogonal_matching_pursuit


def omp(A, b, k):
    # OMP Solve the P0 problem via OMP
    #
    # Solves the following problem:
    #   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
    #
    # The solution is returned in the vector x

    solution = orthogonal_matching_pursuit(A, b=b, n_nonzero_coefs=k,
                                           least_squares=False)
    x = solution.x

    # return the obtained x
    return x
