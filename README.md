# Sparse Representation Algorithms

Python functions to analyze sparse representations by solving the P0- and P1-problems.

1. P0-epsilon problem: `min_x ||x||_0 s.t. ||Ax - b|| < eps`
2. P1-epsilon problem: `min_x ||x||_1 s.t. ||Ax - b|| < eps`

## Algorithms

* Coherence of a matrix:
    * Mutual Coherence
    * Babbel Function
    * Spark of a matrix
* Greedy algorithms, approximating the P0-problem:
    * Orthogonal Matching Pursuit (OMP)
    * Least Squares OMP (LS-OMP)
    * Matching Pursuit (MP)
    * Weak Matching Pursuit (WMP)
    * Thresholding Algorithm (THR)
* Relaxation algorithms, approximating the P0-problem:
    * Basis Pursuit (L1-relaxation)
    * Basis Pursuit + ADMM
    * Iterative Shrinkage Algorithm (ISTA, Fast ISTA)
    * Learned Iterative Shrinkage-Thresholding Algorithm (LISTA)

`sparse.nn` module contains PyTorch implementation of Basis Pursuit & LISTA methods (see [examples](sparse/examples)).


## Documentation

See https://sparse-representation.readthedocs.io

The theoretical foundations of these functions are in edX course
[Sparse Representations in Signal and Image Processing](
https://courses.edx.org/courses/course-v1:IsraelX+236862.1x+3T2019/course/).


## Installation

```
$ git clone https://github.com/dizcza/sparse-representation.git
$ cd sparse-representation
$ pip install -e .
```

### NN module

If you want to install `sparse.nn` module, run `pip install -e .[extra]`.

Before running any examples, start visdom server with 

```
$ python -m visdom.server
```

Then proceed to the examples.

More examples are at http://85.217.171.57:8097. Choose environments with `MatchingPursuit`.
