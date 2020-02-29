# In this project we demonstrate the OMP and BP algorithms, by running them 
# on a set of signals and checking whether they provide the desired outcome
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from midproj.lp import lp
from midproj.omp import omp

# %% Parameters

REPORT_DIR = Path(__file__).parent / "report"
REPORT_DIR.mkdir(exist_ok=True)

# Set the length of the signal
n_signal = 50

# Set the number of atoms in the dictionary
m_atoms = 100

# Set the maximum number of non-zeros in the generated vector
s_max = 20

# Set the minimal entry value
min_coeff_val = -1.

# Set the maximal entry value
max_coeff_val = 1.

# Number of realizations
num_realizations = 200

# Base seed: A non-negative integer used to reproduce the results
base_seed = 28

# %% Create the dictionary

# Create a random matrix A of size (n x m)
A = np.random.randn(n_signal, m_atoms).astype(np.float32)

# Normalize the columns of the matrix to have a unit norm
A_normalized = A / np.linalg.norm(A, axis=0)

# %% Create data and run OMP and BP

# Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4
# Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4

# Allocate a matrix to save the L2 error of the obtained solutions
L2_error = np.zeros((s_max, num_realizations, 2), dtype=np.float32)
# Allocate a matrix to save the support recovery score
support_error = np.zeros((s_max, num_realizations, 2), dtype=np.float32)

# Loop over the sparsity level
for s in trange(s_max, desc="Loop over the sparsity level"):

    s = s + 1
    # Use the same random seed in order to reproduce the results if needed
    np.random.seed(s + base_seed)

    # Loop over the number of realizations
    for experiment in range(num_realizations):
        # In this part we will generate a test signal b = A_normalized @ x by
        # drawing at random a sparse vector x with s non-zeros entries in 
        # true_supp locations with values in the range of
        # [min_coeff_val, max_coeff_val]

        x = np.zeros(m_atoms, dtype=np.float32)

        # Draw at random a true_supp vector
        true_supp = np.random.random_sample(size=s).astype(np.float32)

        # Draw at random the coefficients of x in true_supp locations
        nonzero_ids = np.random.randint(low=0, high=m_atoms, size=s)
        x[nonzero_ids] = true_supp

        def get_support_error(est_support):
            intersection = set(est_support).intersection(nonzero_ids)
            denom = max(s, len(est_support))
            return 1. - len(intersection) / denom

        # Create the signal b
        b = A_normalized.dot(x)

        # Run OMP
        x_omp = omp(A_normalized, b=b, k=s)

        # Compute the relative L2 error
        x_norm = np.linalg.norm(x)
        L2_error[s-1, experiment, 0] = np.linalg.norm(x - x_omp) / x_norm

        # Get the indices of the estimated support
        estimated_supp_omp = np.nonzero(x_omp)[0]

        # Compute the support recovery error
        support_error[s-1, experiment, 0] = get_support_error(
            estimated_supp_omp)

        # Run BP
        x_lp = lp(A_normalized, b=b, tol=tol_lp)

        # Compute the relative L2 error
        L2_error[s-1, experiment, 1] = np.linalg.norm(x - x_lp) / x_norm

        # Get the indices of the estimated support, where the
        # coeffecients are larger (in absolute value) than eps_coeff
        estimated_supp_lp = np.nonzero(np.abs(x_lp) > eps_coeff)[0]

        # Compute the support recovery score
        # Write your code here... support_error[s-1,experiment,1] = ????
        support_error[s-1, experiment, 1] = get_support_error(
            estimated_supp_lp)

# %% Display the results
plt.rcParams.update({'font.size': 14})
# Plot the average relative L2 error, obtained by the OMP and BP versus the
# cardinality
plt.figure()
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 0], axis=1),
         color='red')
plt.plot(np.arange(s_max) + 1, np.mean(L2_error[:s_max, :, 1], axis=1),
         color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Average and Relative L2-Error')
plt.axis((0, s_max, 0, 1))
plt.legend(['OMP', 'LP'])
plt.savefig(REPORT_DIR / "L2-error.png")
plt.show()

# Plot the average support recovery score, obtained by the OMP and BP versus
# the cardinality
plt.figure()
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 0], axis=1),
         color='red')
plt.plot(np.arange(s_max) + 1, np.mean(support_error[:s_max, :, 1], axis=1),
         color='green')
plt.xlabel('Cardinality of the true solution')
plt.ylabel('Probability of Error in Support')
plt.axis((0, s_max, 0, 1))
plt.legend(['OMP', 'LP'])
plt.savefig(REPORT_DIR / "support-error.png")
plt.show()
