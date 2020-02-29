from collections import namedtuple
from functools import wraps

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import orthogonal_mp

Solution = namedtuple("Solution", ("x", "support", "residuals"))


def _trim_atoms(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        solution = func(*args, **kwargs)
        assert isinstance(solution, Solution)
        # residuals' shape is (n_iters, x_dim)
        residuals_norm = np.linalg.norm(solution.residuals, axis=1)
        early_stop = residuals_norm.argmin() + 1
        # If the algorithm is MP or WMP (func is matching_pursuit),
        # the same atom might appear more than once. But it does not break
        # the logic since the expression
        # x[[..., i,i]] = solution.x[[..., i,i]] preserves the valid atoms.
        atoms = solution.support[: early_stop]
        x = np.zeros_like(solution.x)
        x[atoms] = solution.x[atoms]
        residuals = solution.residuals[: early_stop]
        solution = Solution(x, atoms, residuals)
        return solution

    return wrapped


@_trim_atoms
def orthogonal_matching_pursuit(mat_a, b, n_nonzero_coefs,
                                least_squares=False,
                                tol=1e-6):
    r"""
    Given the :math:`\text{P}_0` problem of a system of linear equations

    .. math::
        \min_x{ \| x \|_0} \quad \text{s.t.} \ \boldsymbol{A} \vec{x} = \vec{b}
        :label: p0

    orthogonal Matching Pursuit (OMP) algorithm finds an approximate sparsest
    solution :math:`\vec{x}_{\text{sparsest}}` to :eq:`p0` by solving

    .. math::
        \min_x{ \|\vec{b} - \boldsymbol{A} \vec{x} \|_2^2} \quad \text{s.t.}
        \ \|x\|_0 \le k
        :label: eq_constrained

    where :math:`k` is the maximum number of non-zero real-valued coefficients
    (atoms) of :math:`\vec{x}`.

    Parameters
    ----------
    mat_a : (N, M) np.ndarray
        A fixed weight matrix :math:`\boldsymbol{A}` in the equation
        :eq:`eq_constrained`.
    b : (M,) np.ndarray
        The right side of the equation :eq:`eq_constrained`.
    n_nonzero_coefs : int
        :math:`k`, the maximum number of non-zero coefficients in
        :math:`\vec{x}`.
    least_squares : bool, optional
        If set to True, uses Least Squares OMP (LS-OMP) method instead of OMP
        (False).
        Default is False (OMP).
    tol : float, optional
        The tolerance which determines when a solution is “close enough” to
        the optimal solution. Compared with L2-norm of residuals (errors) at
        each iteration.

    Returns
    -------
    Solution:
        A `namedtuple` with the following attributes:
            `.x` - :math:`\vec{x}_{\text{sparsest}}` solution of
            :eq:`eq_constrained`

            `.support` - a list of atoms (non-zero elements of :math:`\vec{x}`)

            `.residuals` - a list of residual vectors after each iteration

    Notes
    -----
    If the true unknown :math:`\vec{x}_{\text{true}}` to be found is sparse
    enough,

    .. math::
        \|\vec{x}_{\text{true}}\|_0 = s < \frac{1}{2} \left( 1 +
        \frac{1}{\mu\{ \boldsymbol{A} \}} \right)
        :label: sparseness_cond

    where :math:`\mu\{ \boldsymbol{A} \}` is the mutual coherence of the input
    matrix `mat_a` (see :func:`sparse.coherence.mutual_coherence`),
    then WMP, MP, OMP, and LS-OMP are guaranteed to find it.

    """
    # column norms of matrix A
    mat_a_norms = np.linalg.norm(mat_a, axis=0)
    mat_a = mat_a / mat_a_norms
    support = []
    x_solution = np.zeros(shape=mat_a.shape[1], dtype=np.float32)
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
        if np.linalg.norm(residuals) < tol:
            # residuals is a zero vector
            break
        if least_squares:
            # LS-OMP method
            errors = np.full_like(x_solution, np.inf)
            for col_id in set(range(mat_a.shape[1])).difference(support):
                support_cand = support + [col_id]
                x_cand = np.copy(x_solution)
                _, residuals = update_step(x_cand, support_cand)
                errors[col_id] = np.linalg.norm(residuals)
        else:
            # OMP method:
            # min_ai  ||r_i||^2 - (a_i @ r_i / ||a_i||)^2
            # for each atom a_i is equivalent to
            # max_ai  a_i @ r_i / ||a_i||
            # ||a_i|| == 1 since we normalize matrix A
            errors = mat_a.T.dot(residuals)
            errors = -errors
        atom = errors.argmin()

        assert atom not in support, "Each atom should be taken only once by " \
                                    "design of the algorithm ('orthogonal')."
        support.append(atom)
        x_solution, residuals = update_step(x_solution, support)
        residuals_history.append(residuals)
    x_solution /= mat_a_norms
    return Solution(x_solution, support, residuals_history)


@_trim_atoms
def matching_pursuit(mat_a, b, n_iters, weak_threshold=1., tol=1e-9):
    r"""
    (Weak) Matching Pursuit (MP, WMP) algorithm of finding an approximate
    sparsest solution :math:`\vec{x}_{\text{sparsest}}` of the system of
    linear equations :eq:`eq_constrained`.

    Parameters
    ----------
    mat_a : (N, M) np.ndarray
        A fixed weight matrix :math:`\boldsymbol{A}` in the equation
        :eq:`eq_constrained`.
    b : (M,) np.ndarray
        The right side of the equation :eq:`eq_constrained`.
    n_iters : int
        The number of iterations to perform.
        The number of non-zero coefficients in the solution :math:`\vec{x}` is
        at most `n_iters`.
    weak_threshold : float, optional
        A threshold in range (0, 1] for WMP algorithm that defines an early
        stop. If set to `1.`, MP algorithm is used.
        Default is 1. (MP).
    tol : float, optional
        The tolerance which determines when a solution is “close enough” to
        the optimal solution. Compared with L2-norm of residuals (errors) at
        each iteration.

    Returns
    -------
    Solution:
        The solution of :eq:`eq_constrained`. Refer to the output
        documentation of :func:`orthogonal_matching_pursuit`.

    Notes
    -----
    If the sparseness condition :eq:`sparseness_cond` is satisfied,
    the true unknown solution is recovered.

    """
    mat_a_norms = np.linalg.norm(mat_a, axis=0)
    mat_a = mat_a / mat_a_norms
    support = []
    x_solution = np.zeros(shape=mat_a.shape[1], dtype=np.float32)
    residuals = np.copy(b)
    residuals_history = []

    for iter_id in range(n_iters):
        residuals_norm = np.linalg.norm(residuals)
        if residuals_norm < tol:
            break
        errors_maximize = mat_a.T.dot(residuals)
        if weak_threshold < 1:
            # Weak Matching Pursuit
            early_stop = np.nonzero(
                errors_maximize >= weak_threshold * residuals_norm)[0]
            if len(early_stop) > 0:
                early_stop = early_stop[0] + 1
                errors_maximize = errors_maximize[:early_stop]
        atom = errors_maximize.argmax()
        support.append(atom)
        x_solution[atom] += mat_a[:, atom].dot(residuals)
        residuals = b - mat_a.dot(x_solution)
        residuals_history.append(residuals)
    x_solution /= mat_a_norms
    return Solution(x_solution, support, residuals_history)


@_trim_atoms
def thresholding_algorithm(mat_a, b, n_nonzero_coefs):
    r"""
    Thresholding algorithm of finding an approximate
    sparsest solution :math:`\vec{x}_{\text{sparsest}}` of the system of
    linear equations :eq:`eq_constrained`.

    Parameters
    ----------
    mat_a : (N, M) np.ndarray
        A fixed weight matrix :math:`\boldsymbol{A}` in the equation
        :eq:`eq_constrained`.
    b : (M,) np.ndarray
        The right side of the equation :eq:`eq_constrained`.
    n_nonzero_coefs : int
        :math:`k`, the maximum number of non-zero coefficients in
        :math:`\vec{x}`.

    Returns
    -------
    Solution:
        The solution of :eq:`eq_constrained`. Refer to the output
        documentation of :func:`orthogonal_matching_pursuit`.

    Notes
    -----
    If the true unknown :math:`\vec{x}_{\text{true}}` to be found is sparse
    enough,

    .. math::
        \|\vec{x}_{\text{true}}\|_0 = s < \frac{1}{2} \left( 1 +
        \frac{\mid x_{min} \mid}{\mid x_{max} \mid} \cdot
        \frac{1}{\mu\{ \boldsymbol{A} \}} \right)
        :label: sparseness_cond_thr

    where :math:`\mu\{ \boldsymbol{A} \}` is the mutual coherence of the input
    matrix `mat_a` (see :func:`sparse.coherence.mutual_coherence`),
    then the Thresholding algorithm is guaranteed to find it.

    """
    assert n_nonzero_coefs <= mat_a.shape[1], \
        "Does not make sense to use fast method and yet sweep through all " \
        "columns. Use any other algorithm."
    mat_a_norms = np.linalg.norm(mat_a, axis=0)
    mat_a = mat_a / mat_a_norms
    beta = np.abs(mat_a.T.dot(b))
    atoms_sorted = np.argsort(beta)[::-1]
    x_solution = np.zeros(shape=mat_a.shape[1], dtype=np.float32)
    residuals_history = []

    for iter_id in range(1, n_nonzero_coefs + 1):
        support = atoms_sorted[:iter_id]
        a_support = np.take(mat_a, support, axis=1)
        a_inv = np.linalg.pinv(a_support)
        x_solution[support] = a_inv.dot(b)
        residuals = b - mat_a.dot(x_solution)
        residuals_history.append(residuals)
    support = atoms_sorted[:n_nonzero_coefs]
    x_solution /= mat_a_norms
    return Solution(x_solution, support, residuals_history)


def _describe(solution: Solution, method_desc=''):
    residuals_norm = np.linalg.norm(solution.residuals, axis=1)
    print(f"\n{method_desc} solution: {solution.x}"
          f"\n atoms chosen: {solution.support},"
          f"\n residuals norm: {residuals_norm},"
          f"\n residuals: \n{solution.residuals}")


def _quiz5():
    mat_a = [0.1817, 0.5394, -0.1197, 0.6404, 0.6198, 0.1994, 0.0946, -0.3121,
             -0.7634, -0.8181, 0.9883, 0.7018]
    mat_a = np.reshape(mat_a, (3, 4))
    mat_a /= np.linalg.norm(mat_a, axis=0)
    b = np.array([1.1862, -0.1158, -0.1093])
    n_nonzero_coefs = 2

    x_sklearn = orthogonal_mp(mat_a, b, n_nonzero_coefs=n_nonzero_coefs,
                              return_path=False)
    print(f"sklearn solution: {x_sklearn}, "
          f"residual norm: {np.linalg.norm(b - mat_a.dot(x_sklearn))}")
    for least_squares in (False, True):
        solution = orthogonal_matching_pursuit(mat_a, b,
                                               n_nonzero_coefs=n_nonzero_coefs,
                                               least_squares=least_squares)
        _describe(solution, method_desc="LS-OMP" if least_squares else "OMP")
        assert_array_almost_equal(x_sklearn, solution.x)
    solution_mp = matching_pursuit(mat_a, b, n_iters=n_nonzero_coefs,
                                   weak_threshold=0.5)
    _describe(solution_mp, method_desc="WMP(thr=0.5)")
    solution_thr = thresholding_algorithm(mat_a, b, n_nonzero_coefs=3)
    _describe(solution_thr, method_desc="Thresholding")


if __name__ == '__main__':
    _quiz5()
