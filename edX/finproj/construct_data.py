import numpy as np


def construct_data(A, p, sigma, k):
    # CONSTRUCT_DATA Generate Mondrian-like synthetic image
    #
    # Input:
    #  A     - Dictionary of size (n**2 x m)
    #  p     - Percentage of known data in the range (0 1]
    #  sigma - Noise std
    #  k     - Cardinality of the representation of the synthetic image in the
    #          range [1 max(m,n)]
    #
    # Output:
    #  x0 - A sparse vector creating the Mondrian-like image b0 of length m
    #  b0 - The original image of size n^2
    #  noise_std  - The standard deviation of the noise added to b0
    #  b0_noisy   - A noisy version of b0 of size n^2
    #  C  - Sampling matrix of size (p*(n**2) x n^2), 0 < p <= 1
    #  b  - The corrupted image (noisy and subsampled version of b0) of size
    #  p*(n**2)

    # Get the size of the image and number of atoms
    n_squared, m = np.shape(A)
    n = int(np.sqrt(n_squared))

    # generate a Mondrian image
    #  by drawing at random a sparse vector x0 of length m with cardinality k

    # TODO: Draw at random the locations of the non-zeros
    # Write your code here... nnz_locs = ????
    nnz_locs = np.random.choice(m, size=k, replace=False)

    # TODO: Draw at random the values of the coefficients
    # Write your code here... nnz_vals = ????
    nnz_vals = np.random.randn(k)

    # TODO: Create a k-sparse vector x0 of length m given the nnz_locs and nnz_vals
    # Write your code here... x0 = ????
    x0 = np.zeros(m, dtype=np.float32)
    x0[nnz_locs] = nnz_vals

    # TODO: Given A and x0, compute the signal b0
    # Write your code here... b0 = ????
    b0 = A.dot(x0)

    # Create the measured data vector b of size n^2

    # TODO: Compute the dynamic range
    # Write your code here... dynamic_range = ????
    dynamic_range = b0.max() - b0.min()

    # Create a noise vector
    noise_std = sigma * dynamic_range
    noise = noise_std * np.random.randn(n ** 2).astype(np.float32)

    # TODO: Add noise to the original image
    # Write your code here... b0_noisy = ????
    b0_noisy = b0 + noise

    # Create the sampling matrix C of size (p*n^2 x n^2), 0 < p <= 1

    # TODO: Create an identity matrix of size (n^2 x n^2)
    # Write your code here... I = ????
    I = np.eye(n_squared, n_squared, dtype=np.float32)

    # TODO: Draw at random the indices of rows to be kept
    # Write your code here... keep_inds = ????
    b_size = int(p * n_squared)
    keep_inds = np.random.choice(n_squared, size=b_size, replace=False)

    # TODO: Create the sampling matrix C of size (p*n^2 x n^2) by keeping rows
    # from I that correspond to keep_inds
    # Write your code here... C = ????
    C = I[keep_inds]

    # TODO: Create a subsampled version of the noisy image
    # Write your code here... b = ????
    b = C.dot(b0_noisy)

    return x0, b0, noise_std, b0_noisy, C, b
