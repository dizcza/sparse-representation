# Sparse Representation

Python functions to analyze sparse representation by solving P0-problem: `min_x ||x||_0  s.t. Ax = b`,

* Coherence of a matrix:
    * Mutual Coherence
    * Babbel Function
    * Spark of a matrix
* Greedy algorithms, approximating P0-problem:
    * Orthogonal Matching Pursuit (OMP)
    * Least Squares OMP (LS-OMP)
    * Matching Pursuit (MP)
    * Weak Matching Pursuit (WMP)
    * Thresholding Algorithm (THR)
* Relaxation algorithms, approximating P0-problem:
    * Basis Pursuit (L1-relaxation)
    * Basis Pursuit + ADMM

The theoretical foundation of these functions is in edX course
[Sparse Representations in Signal and Image Processing](
https://courses.edx.org/courses/course-v1:IsraelX+236862.1x+3T2019/course/).
