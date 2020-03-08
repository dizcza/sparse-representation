import numpy as np


def soft_shrinkage(x, threshold):
    x_soft = np.zeros_like(x)  # 0 if |x| < lmbda
    mask_less = x <= -threshold
    mask_greater = x >= threshold
    x_soft[mask_less] = x[mask_less] + threshold
    x_soft[mask_greater] = x[mask_greater] - threshold
    return x_soft
