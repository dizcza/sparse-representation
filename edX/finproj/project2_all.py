from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from edX.finproj.bp_admm import bp_admm
from edX.finproj.compute_effective_dictionary import compute_effective_dictionary
from edX.finproj.compute_psnr import compute_psnr
from edX.finproj.construct_data import construct_data
from edX.finproj.oracle import oracle
from edX.midproj.omp import omp

# In this project we will solve a variant of the P0^\epsilon for filling-in 
# missing pixels (also known as "inpainting") in a synthetic image.

# %% Parameters

REPORT_DIR = Path(__file__).parent / "report"
REPORT_DIR.mkdir(exist_ok=True)

# TODO: Set the size of the desired image is (n x n)
# Write your code here... n = ????
n_size = 40


# TODO: Set the number of atoms
# Write your code here... m = ????
m_atoms = 2 * n_size ** 2

# TODO: Set the percentage of known data
# Write your code here... p = ????
p = 0.4

# TODO: Set the noise std
# Write your code here... sigma = ????
sigma = 0.05

# TODO: Set the cardinality of the representation
# Write your code here... true_k = ????
true_k = 10

# Base seed - A non-negative integer used to reproduce the results
# TODO: Set an arbitrary value for base_seed
# Write your code here... base_seed = ????
base_seed = 28

# Run the different algorithms for num_experiments and average the results
num_experiments = 10

# %% Create a dictionary A of size (n**2 x m) for Mondrian-like images

# TODO: initialize A with zeros
# Write your code here... A = ????
A = np.zeros((n_size ** 2, m_atoms), dtype=np.float32)


# In this part we construct A by creating its atoms one by one, where
# each atom is a rectangle of random size (in the range 5-20 pixels),
# position (uniformly spread in the area of the image), and sign. 
# Lastly we will normalize each atom to a unit norm.
for i in range(A.shape[1]):

    # Choose a specific random seed to reproduce the results
    np.random.seed(i + base_seed)

    empty_atom_flag = 1

    while empty_atom_flag:

        # TODO: Create a rectangle of random size and position
        # Write your code here... atom = ????
        atom = np.zeros((n_size, n_size), dtype=np.float32)
        rect_h, rect_w = np.random.randint(5, 20, size=2)
        rect_y, rect_x = np.random.randint(
            0, (n_size - rect_h, n_size - rect_w), size=2)
        sign = np.random.choice((-1, 1))
        atom[rect_y: rect_y + rect_h, rect_x: rect_x + rect_w] = sign

        # Reshape the atom to a 1D vector
        atom = np.reshape(atom, (-1))

        # Verify that the atom is not empty or nearly so
        if np.linalg.norm(atom) > 1e-5:
            empty_atom_flag = 0

            # TODO: Normalize the atom
            # Write your code here... atom = ????
            atom = atom / np.linalg.norm(atom)

            # Assign the generated atom to the matrix A
            A[:, i] = atom


def display_mondorian(k_estimated):
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p=p, sigma=sigma,
                                                         k=true_k)
    plt.figure(figsize=(10, 2))
    plt.subplot(231)
    plt.imshow(b0.reshape((n_size, n_size)))
    plt.title(f"Orig. b0 [k_true={true_k}]")
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(b0_noisy.reshape((n_size, n_size)))
    plt.title("b0_noisy")
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(C.T.dot(b).reshape((n_size, n_size)))
    plt.title("b corrupted")
    plt.axis('off')

    # Compute the OMP estimation
    # omp handles unnormalized (non-unit norm) matrix A
    x_omp = omp(C.dot(A), b, k=k_estimated)

    # Compute the estimated image
    b_omp = A @ x_omp

    plt.subplot(234)
    plt.imshow(b_omp.reshape((n_size, n_size)))
    plt.title(f"b-OMP reconstructed [k={k_estimated}]")
    plt.axis('off')

    from sparse.relaxation import ista
    b_ista = ista(C.dot(A), b, lambd=0.05)
    b_ista = A @ b_ista
    plt.subplot(235)
    plt.imshow(b_ista.reshape((n_size, n_size)))
    plt.title("ISTA")
    plt.axis('off')

    b_bmp = bp_admm(C.dot(A), b, lmbda=0.05)
    b_bmp = A @ b_bmp
    plt.subplot(236)
    plt.imshow(b_bmp.reshape((n_size, n_size)))
    plt.title("BP ADMM")
    plt.axis('off')

    plt.savefig(REPORT_DIR / "reconstructed.png")
    plt.show()


# %% Oracle Inpainting

# Allocate a vector to store the PSNR results
PSNR_oracle = np.zeros(num_experiments, dtype=np.float32)

# Loop over num_experiments
for experiment in range(num_experiments):
    # Choose a specific random seed to reproduce the results
    np.random.seed(experiment + base_seed)

    # Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

    # TODO: Compute the subsampled dictionary
    # Write your code here... A_eff = ????
    A_eff = C.dot(A)

    # TODO: Compute the oracle estimation
    # Write your code here... x_oracle = oracle(?,?,?)
    x_oracle = oracle(A_eff, b, s=np.nonzero(x0)[0])

    # Compute the estimated image of size n^2
    b_oracle = A @ x_oracle

    # Compute the PSNR
    PSNR_oracle[experiment] = compute_psnr(b0, b_oracle)

    # Print some statistics
    print(f'Oracle experiment {experiment+1}/{num_experiments}, '
          f'PSNR: {PSNR_oracle[experiment]:.3f}')


# Display the average PSNR of the oracle
print('Oracle: Average PSNR = %.3f\n' % np.mean(PSNR_oracle))

# %% Greedy: OMP Inpainting

# We will sweep over k = 1 up-to k = max_k and pick the best result
max_k = min(2 * true_k, m_atoms)

# Allocate a vector to store the PSNR estimations per each k
PSNR_omp = np.zeros((num_experiments, max_k))

# Loop over the different realizations
for experiment in range(num_experiments):

    # Choose a specific random seed to reproduce the results
    np.random.seed(experiment + 1 + base_seed)

    # Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

    # Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)

    # Run the OMP for various values of k and pick the best results
    for k_ind in range(max_k):

        # Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, k_ind + 1)

        # Un-normalize the coefficients
        x_omp = x_omp / atoms_norm

        # Compute the estimated image        
        b_omp = A @ x_omp

        # Compute the current PSNR
        PSNR_omp[experiment, k_ind] = compute_psnr(b0, b_omp)

        # Save the best result of this realization, we will present it later
        if PSNR_omp[experiment, k_ind] == max(PSNR_omp[experiment, :]):
            best_b_omp = b_omp

        # Print some statistics
        print(f'OMP experiment {experiment+1}/{num_experiments}, '
              f'cardinality {k_ind}, '
              f'PSNR: {PSNR_omp[experiment, k_ind]:.3f}')

# Compute the best PSNR, computed for different values of k
PSNR_omp_best_k = np.max(PSNR_omp, axis=-1)

# Display the average PSNR of the OMP (obtained by the best k per image)
print('OMP: Average PSNR = %.3f\n' % np.mean(PSNR_omp_best_k))

# Plot the average PSNR vs. k
psnr_omp_k = np.mean(PSNR_omp, 0)
k_scope = np.arange(1, max_k + 1)
plt.figure(1)
plt.plot(k_scope, psnr_omp_k, '-r*')
plt.xlabel("k", fontsize=16)
plt.ylabel("PSNR [dB]", fontsize=16)
plt.title(f"OMP: PSNR vs. k, True Cardinality = {true_k}")
plt.vlines(x=true_k, ymin=psnr_omp_k.min(), ymax=psnr_omp_k.max(),
           linestyles='--')
plt.savefig(REPORT_DIR / "OMP-psnr.png")
plt.show()

k_highest_psnr = psnr_omp_k.argmax() + 1
print(f"argmax_k PSNR(k) = {k_highest_psnr}")
display_mondorian(k_estimated=k_highest_psnr)

# %% Convex relaxation: Basis Pursuit Inpainting via ADMM

# We will sweep over various values of lambda
num_lambda_values = 10

# Allocate a vector to store the PSNR results obtained for the best lambda
PSNR_admm_best_lambda = np.zeros(num_experiments)

# Loop over num_experiments
for experiment in range(num_experiments):

    # Choose a specific random seed to reproduce the results
    np.random.seed(experiment + 1 + base_seed)

    # Construct data
    [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

    x_oracle = oracle(C.dot(A), b, s=np.nonzero(x0)[0])
    b_oracle = A @ x_oracle

    # Compute the effective subsampled dictionary
    [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)

    # Run the BP for various values of lambda and pick the best result
    lambda_max = np.linalg.norm(A_eff_normalized.T @ b, np.inf)
    lambda_vec = np.logspace(-5, 0, num_lambda_values) * lambda_max
    psnr_admm_lambda = np.zeros(num_lambda_values)

    # Loop over various values of lambda
    for lambda_ind in range(num_lambda_values):

        # Compute the BP estimation
        x_admm = bp_admm(A_eff_normalized, b, lambda_vec[lambda_ind])

        # Un-normalize the coefficients
        x_admm = x_admm / atoms_norm

        # Compute the estimated image        
        b_admm = A @ x_admm

        # Compute the current PSNR
        psnr_admm_lambda[lambda_ind] = compute_psnr(b0, b_admm)

        # Save the best result of this realization, we will present it later
        if psnr_admm_lambda[lambda_ind] == max(psnr_admm_lambda):
            best_b_admm = b_admm

        # print some statistics
        print(f'BP experiment {experiment+1}/{num_experiments}, '
              f'lambda {lambda_ind+1}/{num_lambda_values}, '
              f'PSNR {psnr_admm_lambda[lambda_ind]:.3f}')

    # Save the best PSNR
    PSNR_admm_best_lambda[experiment] = max(psnr_admm_lambda)

# Display the average PSNR of the BP
print("BP via ADMM: Average PSNR = ", np.mean(PSNR_admm_best_lambda), "\n")

# Plot the PSNR vs. lambda of the last realization
plt.figure(2)
plt.semilogx(lambda_vec, psnr_admm_lambda, '-*r')
plt.xlabel(r'$\lambda$', fontsize=16)
plt.ylabel("PSNR [dB]", fontsize=16)
plt.title(r"BP via ADMM: PSNR vs. $\lambda$")
plt.savefig(REPORT_DIR / "BP-ADMM-psnr.png")

# %% show the results

# Show the images obtained in the last realization, along with their PSNR
plt.figure(3, figsize=(10, 6))

plt.subplot(231)
plt.imshow(np.reshape(b0, (n_size, n_size)), cmap='gray')
plt.title(f"Original Image, k = {true_k}")
plt.axis('off')

plt.subplot(232)
plt.imshow(np.reshape(b0_noisy, (n_size, n_size)), cmap='gray')
plt.title(f"Noisy Image, PSNR = {compute_psnr(b0, b0_noisy):.1f}")
plt.axis('off')

plt.subplot(233)
corrupted_img = np.reshape(C.T @ b, (n_size, n_size))
plt.imshow(corrupted_img, cmap='gray')
plt.title(f"Corrupted Image, PSNR = {compute_psnr(b0, C.T @ b):.1f}")
plt.axis('off')

plt.subplot(234)
plt.imshow(np.reshape(b_oracle, (n_size, n_size)), cmap='gray')
plt.title(f"Oracle, PSNR = {compute_psnr(b0, b_oracle):.1f}")
plt.axis('off')

plt.subplot(235)
plt.imshow(np.reshape(best_b_omp, (n_size, n_size)), cmap='gray')
plt.title(f"OMP, PSNR = {compute_psnr(b0, best_b_omp):.1f}")
plt.axis('off')

plt.subplot(236)
plt.imshow(np.reshape(best_b_admm, (n_size, n_size)), cmap='gray')
plt.title(f"BP-ADMM, PSNR = {compute_psnr(b0, best_b_admm):.1f}")
plt.axis('off')
plt.savefig(REPORT_DIR / "reconstructed-best.png")

# %% Compare the results

# Show a bar plot of the average PSNR value obtained per each algorithm
plt.figure(4)
mean_psnr = np.array([np.mean(PSNR_oracle), np.mean(PSNR_omp_best_k),
                      np.mean(PSNR_admm_best_lambda)])
x_bar = np.arange(3)
plt.bar(x_bar, mean_psnr)
plt.xticks(x_bar, ('Oracle', 'OMP', 'BP-ADMM'))
plt.ylabel('PSNR [dB]', fontsize=16)
plt.xlabel('Algorithm', fontsize=16)
plt.savefig(REPORT_DIR / "psnr-hist.png")
plt.show()

# %% Run OMP with fixed cardinality and increased percentage of known data

# TODO: Set the noise std
# Write your code here... sigma = ????
sigma = 0.05


# TODO: Set the cardinality of the representation
# Write your code here... true_k = ????
true_k = 5


# TODO: Create a vector of increasing values of p in the range [0.4 1]. The
# length of this vector equal to num_values_of_p = 7.
# Write your code here... num_values_of_p = ???? p_vec = ????
num_values_of_p = 7
p_vec = np.linspace(0.4, 1.0, num=num_values_of_p, endpoint=True,
                    dtype=np.float32)

# We will repeat the experiment for num_experiments realizations
num_experiments = 100

# Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_p = np.zeros(num_values_of_p, dtype=np.float32)

# Loop over num_experiments
for experiment in range(num_experiments):

    # Loop over various values of p
    for p_ind in range(num_values_of_p):
        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment + 1 + base_seed)

        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p_vec[p_ind],
                                                             sigma, true_k)

        # Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)

        # Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k)

        # Un-normalize the coefficients
        x_omp = x_omp / atoms_norm

        # Compute the estimated image        
        b_omp = A @ x_omp

        # TODO: Compute the MSE of the estimate
        # Write your code here... cur_mse = ????
        cur_mse = np.mean((b0 - b_omp) ** 2)
        dynamic_range = b0.max() - b0.min()

        # Compute the current normalized MSE and aggregate
        mse_omp_p[p_ind] = mse_omp_p[p_ind] + cur_mse / (
                    dynamic_range * noise_std) ** 2

        # print some statistics
        print(f'OMP as a function p: experiment '
              f'{experiment+1}/{num_experiments}, '
              f'p_ind {p_ind+1}/{num_values_of_p}, '
              f'MSE: {mse_omp_p[p_ind]:.3f}')

# Compute the average PSNR over the different realizations
mse_omp_p = mse_omp_p / num_experiments

# Plot the average normalized MSE vs. p
plt.figure(5)
plt.plot(p_vec, mse_omp_p, '-*r')
plt.ylabel('Normalized-MSE', fontsize=16)
plt.xlabel('p', fontsize=16)
plt.title(f'OMP with k = {true_k}, Normalized-MSE vs. p')
plt.savefig(REPORT_DIR / "OMP-MSE-p.png")
plt.show()

# %% Run OMP with fixed cardinality and increased noise level

# TODO: Set the cardinality of the representation
# Write your code here... true_k = ????
true_k = 5

# TODO: Set the percentage of known data
# Write your code here... p = ????
p = 0.5

# TODO: Create a vector of increasing values of sigma in the range [0.15 0.5].
# The length of this vector equal to num_values_of_sigma = 10.
# Write your code here... num_values_of_sigma = ???? sigma_vec = ????
num_values_of_sigma = 10
sigma_vec = np.linspace(0.15, 0.5, num=num_values_of_sigma, endpoint=True,
                        dtype=np.float32)

# Repeat the experiment for num_experiments realizations
num_experiments = 100

# Run OMP and store the normalized MSE results, averaged over num_experiments
mse_omp_sigma = np.zeros(num_values_of_sigma, dtype=np.float32)

# Loop over num_experiments
for experiment in range(num_experiments):

    # Loop over increasing noise level
    for sigma_ind in range(num_values_of_sigma):
        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment + 1 + base_seed)

        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma_vec[
            sigma_ind], true_k)

        # Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)

        # Compute the OMP estimation
        x_omp = omp(A_eff_normalized, b, true_k)

        # Un-normalize the coefficients
        x_omp = x_omp / atoms_norm

        # Compute the estimated image        
        b_omp = A @ x_omp

        # TODO: Compute the MSE of the estimate
        # Write your code here... cur_mse = ????
        cur_mse = np.mean((b0 - b_omp) ** 2)

        # Compute the current normalized MSE and aggregate
        mse_omp_sigma[sigma_ind] = mse_omp_sigma[sigma_ind] + cur_mse / (
                    noise_std ** 2)

        # Print some statistics
        print(f"OMP as a function sigma: experiment "
              f"{experiment+1}/{num_experiments}, "
              f"sigma_ind {sigma_ind+1}/{num_values_of_sigma}, "
              f"MSE: {mse_omp_sigma[sigma_ind]:.3f}")

# Compute the average PSNR over the different realizations
mse_omp_sigma = mse_omp_sigma / num_experiments

# Plot the average normalized MSE vs. sigma
plt.figure(6)
plt.plot(sigma_vec, mse_omp_sigma, '-*r')
plt.ylim(0.5 * min(mse_omp_sigma), 5 * max(mse_omp_sigma))
plt.ylabel('Normalized-MSE')
plt.xlabel('sigma')
plt.title(f"OMP with k = {true_k}, Normalized-MSE vs. sigma")
plt.savefig(REPORT_DIR / "OMP-MSE-sigma.png")

plt.show()
