import pandas as pd 
import numpy as np
import sys
from scipy.stats import t as t_coeff
from scipy import stats
from joblib import Parallel, delayed

def progress_bar(start, ii, end):
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" %('#'*round(100*(ii+1-start)/(end-start)),round(100*(ii+1-start)/(end-start))))
    sys.stdout.flush()

def compute_bias_vectorized(msl_diff, ii, verbose=False):
    """Compute uncertainty for a vector of differences up to index ii."""
    subsample_msl_diff = msl_diff[:ii]
    p = stats.shapiro(subsample_msl_diff)
    # Calculate autocorrelation at lag 1 using the full sample
    rho1 = np.corrcoef(msl_diff[:-1], msl_diff[1:])[0, 1]
    # Calculate the number of independent measurements
    n_sample = len(subsample_msl_diff)
    n = max((1 - rho1) / (1 + rho1) * n_sample, 2)
    s = np.std(subsample_msl_diff)
    
    if p[1] > 0.05:  # Non-Gaussian
        alpha = 0.32
        student_coeff = t_coeff.ppf(1 - alpha / 2, n - 1)
        uncertainty = student_coeff * s / np.sqrt(n)
    else:
        uncertainty = s / np.sqrt(n)
    
    if verbose:
        print(f"Shapiro-Wilk p-value: {p[1]:.2f}")
        print(f"Independent measurements: {n:.2f}")
        print(f"Standard deviation: {s:.2f}")
        print(f"Offset uncertainty: {uncertainty:.2f}")
        
    return uncertainty, n

def compute_unc_for_cell(msl_diff, ii):
    """Compute uncertainty for a single grid cell, discarding if any NaNs are present."""
    # Discard grid cell if any NaN is present
    if np.isnan(msl_diff).any() or len(msl_diff) < 3:
        return np.nan  # Skip computation
    
    msl_diff = msl_diff * 1e3 
    return compute_bias_vectorized(msl_diff, ii, verbose=False)[0]

def compute_all_unc(diff_map, ii):
    """Vectorized computation for all grid cells using parallelization."""
    shape = diff_map.shape[1:]
    unc = Parallel(n_jobs=-1)(
        delayed(compute_unc_for_cell)(diff_map[:, i, j].values, ii)
        for i in range(shape[0])
        for j in range(shape[1])
    )
    return np.array(unc).reshape(shape)

def masked_mean_along_axis(arr, axis):
    # If any element is masked along the axis, mask the result    
    mean_values = np.ma.mean(arr, axis=axis)
    mask_along_axis = np.any(np.isnan(arr), axis=axis)
    mean_values[mask_along_axis.values] = np.ma.masked

    return mean_values
