import numpy as np


def compute_effective_dictionary(C, A):
    # COMPUTE_EFFECTIVE_DICTIONARY Computes the subsampled and normalized
    #   dictionary
    #
    # Input:
    #  C     - Sampling matrix of size (p*(n**2) x n^2)
    #  A     - Dictionary of size ((n**2) x m)
    #
    # Output:
    #  A_eff_normalized - The subsampled and normalized dictionary of size
    #  (p*(n**2) x m)
    #  atoms_norm - A vector of length m, containing the norm of each sampled
    #  atom

    # TODO: Compute the subsampled dictionary
    # Write your code here... A_eff = ????
    A_eff = C.dot(A)

    # TODO: Compute the norm of each atom
    # Write your code here...  atoms_norm = ????
    atoms_norm = np.linalg.norm(A_eff, axis=0)
    atoms_norm[atoms_norm == 0] += 1e-9

    # TODO: Normalize the columns of A_eff, avoid division by zero
    # Write your code here...  A_eff_normalized = ????
    A_eff_normalized = A_eff / atoms_norm

    return A_eff_normalized, atoms_norm
