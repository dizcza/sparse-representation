from collections import namedtuple

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import orthogonal_mp

Solution = namedtuple("Solution", ("x", "support", "residuals"))


def orthogonal_matching_pursuit(mat_a, b, n_nonzero_coefs=3, least_squares=False):
    norm = np.linalg.norm(mat_a, axis=0)
    assert_array_almost_equal(norm, 1., err_msg="Input matrix should be L2 normalized (col)")
    support = []
    x_solution = np.zeros(shape=(mat_a.shape[1],), dtype=np.float32)
    residuals = np.copy(b)
    residuals_history = []

    def update_step(x, support):
        x = np.copy(x)
        a_support = np.take(mat_a, support, axis=1)
        a_inv = np.linalg.pinv(a_support)
        x[support] = a_inv.dot(b)
        residuals = b - mat_a.dot(x)
        return x, residuals

    for iter_id in range(n_nonzero_coefs):
        if least_squares:
            # LS-OMP method
            errors = np.full_like(x_solution, np.inf)
            for col_id in set(range(mat_a.shape[1])).difference(support):
                support_cand = support + [col_id]
                x_cand = np.copy(x_solution)
                _, residuals = update_step(x_cand, support_cand)
                errors[col_id] = np.linalg.norm(residuals)
        else:
            # OMP method
            errors = -(mat_a.T.dot(residuals))
        atom = errors.argmin()

        # each atom should be taken only once by design of the algorithm ('orthogonal')
        assert atom not in support
        support.append(atom)
        x_solution, residuals = update_step(x_solution, support)
        residuals_history.append(residuals)
    return Solution(x_solution, support, residuals_history)


def matching_pursuit(mat_a, b, n_iters=3, weak_threshold=1.):
    norm = np.linalg.norm(mat_a, axis=0)
    assert_array_almost_equal(norm, 1., err_msg="Input matrix should be L2 normalized (col)")
    support = []
    x_solution = np.zeros(shape=(mat_a.shape[1],), dtype=np.float32)
    residuals = np.copy(b)
    residuals_history = []

    for iter_id in range(n_iters):
        residuals_norm = np.linalg.norm(residuals)
        errors_maximize = mat_a.T.dot(residuals)
        if weak_threshold < 1:
            # Weak Matching Pursuit
            early_stop = np.nonzero(errors_maximize >= weak_threshold * residuals_norm)[0]
            if len(early_stop) > 0:
                early_stop = early_stop[0] + 1
                errors_maximize = errors_maximize[:early_stop]
        atom = errors_maximize.argmax()
        support.append(atom)
        x_solution[atom] += mat_a[:, atom].dot(residuals)
        residuals = b - mat_a.dot(x_solution)
        residuals_history.append(residuals)
    return Solution(x_solution, support, residuals_history)


def describe(solution: Solution, method_info=''):
    print(f"\nSolution ({method_info}): {solution.x}"
          f"\n atoms chosen: {solution.support},"
          f"\n residuals norm: {list(map(np.linalg.norm, solution.residuals))},"
          f"\n residuals: \n{solution.residuals}")


def quiz5():
    mat_a = [0.1817, 0.5394, -0.1197, 0.6404, 0.6198, 0.1994, 0.0946, -0.3121, -0.7634, -0.8181, 0.9883, 0.7018]
    mat_a = np.reshape(mat_a, (3, 4))
    mat_a /= np.linalg.norm(mat_a, axis=0)
    b = np.array([1.1862, -0.1158, -0.1093])
    n_nonzero_coefs = 2

    x_sklearn = orthogonal_mp(mat_a, b, n_nonzero_coefs=n_nonzero_coefs, return_path=False)
    print(f"sklearn solution: {x_sklearn}, residual norm: {np.linalg.norm(b - mat_a.dot(x_sklearn))}")
    for least_squares in (False, True):
        solution = orthogonal_matching_pursuit(mat_a, b, n_nonzero_coefs=n_nonzero_coefs, least_squares=least_squares)
        describe(solution, method_info=f"least_squares={least_squares}")
        assert_array_almost_equal(x_sklearn, solution.x)
    solution_mp = matching_pursuit(mat_a, b, n_iters=n_nonzero_coefs, weak_threshold=0.5)
    describe(solution_mp)


if __name__ == '__main__':
    quiz5()
