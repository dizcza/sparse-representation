import torch
import torch.nn as nn
import unittest
from numpy.testing import assert_array_almost_equal

from mighty.utils.common import set_seed
from sparse.nn.model import Softshrink, LISTA, MatchingPursuit
from sparse.nn.solver import BasisPursuitADMM


class TestSoftshrink(unittest.TestCase):
    def test_softshink(self):
        set_seed(16)
        lambd = 0.1
        softshrink = Softshrink(n_features=1)
        softshrink.lambd.data[:] = lambd
        softshrink_gt = nn.Softshrink(lambd=lambd)
        tensor = torch.randn(10, 20)
        assert_array_almost_equal(softshrink(tensor).detach(),
                                  softshrink_gt(tensor))


class TestLISTA(unittest.TestCase):
    def test_lista_forward_best(self):
        set_seed(16)
        solver = BasisPursuitADMM()
        in_features = 10
        out_features = 40
        lista = LISTA(in_features=in_features, out_features=out_features,
                      solver=solver)
        mp = MatchingPursuit(in_features=in_features,
                             out_features=out_features, solver=solver)
        tensor = torch.randn(5, in_features)
        with torch.no_grad():
            mp.normalize_weight()
            lista.weight_input.data = mp.weight.data.clone()
            mp_output = mp(tensor)
            lista_output_best = lista.forward_best(tensor)
        assert_array_almost_equal(lista_output_best.latent, mp_output.latent)
        assert_array_almost_equal(lista_output_best.reconstructed,
                                  mp_output.reconstructed)


if __name__ == '__main__':
    unittest.main()
