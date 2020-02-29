import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.linear_model import orthogonal_mp

from sparse.greedy_pursuit import orthogonal_matching_pursuit as omp


class OMPTest(unittest.TestCase):

    def test_quiz5(self):
        mat_a = [0.1817, 0.5394, -0.1197, 0.6404,
                 0.6198, 0.1994, 0.0946, -0.3121,
                 -0.7634, -0.8181, 0.9883, 0.7018]
        mat_a = np.reshape(mat_a, (3, 4))
        mat_a /= np.linalg.norm(mat_a, axis=0)
        b = np.array([1.1862, -0.1158, -0.1093])
        n_nonzero_coefs = 2

        solution = omp(mat_a, b=b, n_nonzero_coefs=n_nonzero_coefs)
        solution_lse = omp(mat_a, b=b, n_nonzero_coefs=n_nonzero_coefs,
                           least_squares=True)
        assert_array_almost_equal(solution.x, solution_lse.x)
        assert_array_equal(sorted(solution.support),
                           sorted(solution_lse.support))

        x_sklearn = orthogonal_mp(mat_a, y=b, n_nonzero_coefs=n_nonzero_coefs,
                                  return_path=False)
        assert_array_almost_equal(solution.x, x_sklearn)
        assert_array_almost_equal(solution.x, [0., 1.0000116, 0., 1.0100027])
        assert_array_equal(sorted(solution.support), (1, 3))

    def test_unnormalized(self):
        np.random.seed(28)
        shape = 40, 70
        n_nonzero = 10
        mat_a = 5 * np.random.randn(*shape)
        mat_a_norm = mat_a / np.linalg.norm(mat_a, axis=0)
        x_true = np.zeros(shape[1], dtype=np.float32)
        support_true = np.random.choice(shape[1], size=n_nonzero,
                                        replace=False)
        x_true[support_true] = 1.
        b_true = mat_a.dot(x_true)
        b_norm = mat_a_norm.dot(x_true)

        solution = omp(mat_a, b=b_true, n_nonzero_coefs=n_nonzero)
        solution_norm = omp(mat_a_norm, b=b_norm, n_nonzero_coefs=n_nonzero)

        # solutions' x might differ but the support should match
        assert_array_equal(sorted(solution.support),
                           sorted(solution_norm.support))


if __name__ == '__main__':
    unittest.main()
