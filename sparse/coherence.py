r"""
Mutual Coherence and Babel Function are the properties of a matrix, used to
estimate the Spark of a matrix, which in turn is used to determine the
optimality of the solution to :math:`\text{P}_0` problem.

Babel Function gives a tighter bound on the Spark of a matrix.

Spark of a matrix :math:`\boldsymbol{A}` is the size of the smallest subset of
linearly dependent columns of :math:`\boldsymbol{A}`.

.. currentmodule:: sparse.coherence

.. autosummary::
   :toctree: toctree/coherence/

   mutual_coherence
   babel

"""
import math
from collections import namedtuple

import numpy as np

CoherenceSpark = namedtuple("CoherenceSpark", ("coherence", "spark"))


def mutual_coherence(mat):
    r"""
    For an arbitrary input matrix :math:`\boldsymbol{A}` of size `N` x `M`, the
    mutual coherence is the maximal absolute inner-product between its
    normalized columns :math:`\{ a_i \mid i=1,2,...,M \}`:

    .. math::
        \mu (\boldsymbol{A}) = \max_{1 \le i < j \le M}
        \frac{\mid a_i^\top a_j \mid}{\|a_i\|_2 \|a_j\|_2}
        :label: coh

    The mutual coherence :math:`\mu` lies in range `[0, 1]`.

    At the same time, the Spark lower bound of a matrix is estimated as

    .. math::
        \text{Spark}(\boldsymbol{A}) \ge 1 + \frac{1}{\mu(\boldsymbol{A})}
        :label: spark

    Parameters
    ----------
    mat : (N, M) np.ndarray
        Matrix :math:`\boldsymbol{A}` in :eq:`coh`.

    Returns
    -------
    CoherenceSpark
        A namedtuple with two attributes:
          `.coherence` - mutual coherence of `mat`;

          `.spark` - Spark lower bound :eq:`spark` of `mat`.

    """
    mat = mat / np.linalg.norm(mat, axis=0)
    gram = np.abs(mat.T.dot(mat))
    np.fill_diagonal(gram, 0)
    mu = gram.max()
    spark = math.ceil(1 + 1 / mu)
    return CoherenceSpark(mu, spark)


def babel(mat):
    r"""
    For an arbitrary input matrix :math:`\boldsymbol{A}` of size `N` x `M` and
    normalized columns :math:`\{ a_i \mid i=1,2,...,M \}`, the Babel-Function
    is defined by

    .. math::
        \mu_1(k) = \max_{\mid \Lambda \mid = k} \left[ \max_{j \notin \Lambda}
        \sum_{i \in \Lambda}{\mid a_i^\top a_j \mid} \right]
        :label: babel

    If :math:`\mu_1(k-1) < 1`, this implies that any set of :math:`k` columns
    from :math:`\boldsymbol{A}` are linearly dependent. In this case, the Spark
    necessarily satisfies

    .. math::
        \text{Spark}(\boldsymbol{A}) > k = 1 + \arg \min_k
        \left({\mu_1(k) > 1}\right)
        :label: spark_babel

    Parameters
    ----------
    mat : (N, M) np.ndarray
        Matrix :math:`\boldsymbol{A}` in :eq:`babel`.

    Returns
    -------
    CoherenceSpark
        A `namedtuple` with two attributes:
          `.coherence` - a list of `M-1` elements of
          :math:`\mu_1(k), \ k=1,2,...,M-1`;

          `.spark` - Spark lower bound :eq:`spark_babel` of `mat`.

    Notes
    -----
    :eq:`spark_babel` is a tighter bound on Spark than :eq:`spark`.

    """
    mat = mat / np.linalg.norm(mat, axis=0)
    gram = np.abs(mat.T.dot(mat))
    # Gram matrix' of L2 normalized matrix entries are in range [0, 1]
    # with 1s on the diagonal
    gram.sort(axis=1)  # sort rows
    gram = gram[:, ::-1]  # in descending order
    gram = gram[:, 1:]  # skip the first column of 1s (diagonal elements)
    gram = gram.cumsum(axis=1)  # cumsum rows
    mu1 = gram.max(axis=0)
    spark = np.nonzero(mu1 > 1)[0][0] + 2
    return CoherenceSpark(mu1, spark)


def _quiz4():
    mat = np.reshape([16, -2, 15, 13, 5, 6, 8, 8, 9, 4, 11, 12, 4, 12, 10, 1],
                     (4, 4))
    print(mutual_coherence(mat))
    print(babel(mat))


if __name__ == '__main__':
    _quiz4()
