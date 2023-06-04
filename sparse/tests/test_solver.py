import torch
import unittest
from numpy.testing import assert_array_almost_equal

from mighty.utils.common import set_seed
from sparse.relaxation.basis_pursuit import basis_pursuit_admm as bp_numpy


try:
    import mighty
    MIGHTY_INSTALLED = True
except ImportError:
    MIGHTY_INSTALLED = False


@unittest.skipUnless(MIGHTY_INSTALLED, reason="pip install pytorch-mighty")
class TestBasisPursuit(unittest.TestCase):

    def test_basis_pursuit_admm_as_numpy(self):
        from sparse.nn.solver import basis_pursuit_admm as bp_pytorch

        set_seed(12)
        n_features, n_atoms = 10, 50
        dictionary = torch.randn(n_features, n_atoms)
        tensor_x = torch.randn(3, n_features)
        lambd = 0.1
        tol_decimal = 4
        tol = 10 ** (-tol_decimal)
        max_iters = 100
        z_pytorch = bp_pytorch(A=dictionary, b=tensor_x, lambd=lambd, tol=tol,
                               max_iters=max_iters).numpy()
        z_numpy = [bp_numpy(A=dictionary.numpy(), b=b, lambd=lambd, tol=tol,
                            max_iters=max_iters) for b in tensor_x]
        assert_array_almost_equal(z_pytorch, z_numpy, decimal=tol_decimal)

    def test_basis_pursuit_admm_convergence(self):
        from sparse.nn.solver import BasisPursuitADMM

        set_seed(12)
        n_features, n_atoms = 10, 50
        dictionary = torch.randn(n_features, n_atoms)
        tensor_x = torch.randn(3, n_features)
        solver = BasisPursuitADMM(lambd=0.1, max_iters=1000)
        solver.save_stats = True
        tensor_z = solver.solve(A=dictionary, b=tensor_x)
        iterations = solver.online['iterations'].get_mean().item()
        self.assertLessEqual(iterations, solver.max_iters)
        dv_norm = solver.online['dv_norm'].get_mean().item()
        self.assertLessEqual(dv_norm, solver.tol)
        x_restored = tensor_z.matmul(dictionary.t())
        assert_array_almost_equal(tensor_x, x_restored, decimal=1)
        psnr = solver.online['psnr'].get_mean().item()
        self.assertGreater(psnr, 41.)  # 41 is taken from the output


if __name__ == '__main__':
    unittest.main()
