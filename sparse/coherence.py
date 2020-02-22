import math
from collections import namedtuple

import numpy as np

CoherenceSpark = namedtuple("CoherenceSpark", ("coherence", "spark"))


def mutual_coherence(mat):
    r"""
    Calculates the mutual coherence and the Spark of the matrix `mat` as the
    maximum element of its normalized Gram matrix.

    Parameters
    ----------
    mat : (M, N) np.ndarray
        A weight matrix :math:`A` in the equation

        .. math::
            A  \vec{x} = \vec{b}


    Returns
    -------
    CoherenceSpark
        `CoherenceSpark` namedtuple with two keys:
          `.coherence` - mutual coherence of `mat`;

          `.spark` - spark lowerbound of `mat`.

    """
    mat = mat / np.linalg.norm(mat, axis=0)
    gram = np.abs(mat.T.dot(mat))
    np.fill_diagonal(gram, 0)
    mu = gram.max()
    spark = math.ceil(1 + 1 / mu)
    return CoherenceSpark(mu, spark)


def babel(mat):
    r"""
    Calculates the Babel Function and the Spark of the matrix `mat`. The Babel
    Function gives a tighter bound on estimation of the Spark of a matrix than
    :func:`mutual_coherence`.

    Parameters
    ----------
    mat : (M, N) np.ndarray
        A weight matrix :math:`A` in the equation

        .. math::
            A  \vec{x} = \vec{b}


    Returns
    -------
    CoherenceSpark
        `CoherenceSpark` namedtuple with two keys:
          `.coherence` - mutual coherence of `mat`;

          `.spark` - spark lowerbound of `mat`.

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
    spark = np.where(mu1 > 1)[0][0] + 2
    return CoherenceSpark(mu1, spark)


def _quiz4():
    mat = np.reshape([16, -2, 15, 13, 5, 6, 8, 8, 9, 4, 11, 12, 4, 12, 10, 1],
                     (4, 4))
    print(mutual_coherence(mat))
    print(babel(mat))


if __name__ == '__main__':
    _quiz4()
