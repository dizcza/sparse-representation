import unittest

import numpy as np

from sparse.relaxation import *


class TestRelaxation(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        n_features, n_atoms = 10, 50
        self.A = np.random.randn(n_features, n_atoms)
        self.x = (np.random.randn(n_atoms) > 0.5).astype(np.int32)
        self.b = self.A @ self.x

    def check_convergence(self, x_solved):
        err = np.linalg.norm(self.A @ x_solved - self.b)
        l1_norm = np.linalg.norm(x_solved, ord=1)
        self.assertLess(err, 0.3)
        self.assertLess(l1_norm, 4.0)

    def test_ista(self):
        x_solved = ista(self.A, self.b, lambd=0.3)
        self.check_convergence(x_solved)

    def test_basis_pursuit_linprog(self):
        x_solved = basis_pursuit_linprog(self.A, self.b)
        self.check_convergence(x_solved)

    def test_basis_pursuit_admm(self):
        x_solved = basis_pursuit_admm(self.A, self.b, lambd=0.3)
        self.check_convergence(x_solved)


if __name__ == '__main__':
    unittest.main()
