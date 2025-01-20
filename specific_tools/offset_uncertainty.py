import pandas as pd 
import numpy as np
import sys
from scipy.stats import t as t_coeff
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar
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

# Model for the interpolation function
def f_1_sqrt_n_model(x, a, b, c): # 1/square_root(n)
    return (a / np.sqrt(x + c)) + b

# Fuction to fit the model on the curves
def fit_curves(curves_to_fit, start):
  curves_fitted = []
  for curve, start_cycle in zip(curves_to_fit, start):
    # Define the abscissa axis over which interpolating the curve
    curve_fit_x = np.arange(start_cycle,
                            len(curve[~np.isnan(curve)])+start_cycle, 1)
    # Perform the interpolation where the uncertainty values exist (not NaN)
    res_fit = curve_fit(f_1_sqrt_n_model, curve_fit_x,
                        curve[~np.isnan(curve)]) # 1/sqrt(n) fit
    a_fit, b_fit, c_fit = res_fit[0][0], res_fit[0][1], res_fit[0][2]

    # Define the abscissa axis for the plot
    curve_fit_x_plot = np.arange(1, 90, 1)
    # Interpolated curve for the plot
    curve_fit_y = f_1_sqrt_n_model(curve_fit_x_plot, a_fit, b_fit, c_fit)
    # Store the results of the interpolation
    curves_fitted.append([curve_fit_x_plot, curve_fit_y, a_fit, b_fit, c_fit])

  return curves_fitted

def compute_required_days_outside_tandem(perf_values, curves_fitted):
  repeat_cycle_in_days = 9.91564 # Number of days in one cycle on the
                                 # J3/S6 reference orbit

  # Compute the number of days corresponding to each performance
  corr_nb_days_outside_tandem_std_main = []
  corr_nb_days_outside_tandem_std_comp = []
  for p_val in perf_values:
    def f_1_sqrt_n_minus_value_unc_main(x):
      p_val_x = (curves_fitted[0][2] /
                 np.sqrt(x + curves_fitted[0][4])) + curves_fitted[0][3]
      return np.abs(p_val_x - p_val)
    corr_nb_days_outside_tandem_std_main.append(
        int(np.ceil(minimize_scalar( # Find the number of cycles
            f_1_sqrt_n_minus_value_unc_main).x * repeat_cycle_in_days)))

    def f_1_sqrt_n_minus_value_unc_comp(x):
      p_val_x = (curves_fitted[1][2] /
                 np.sqrt(x + curves_fitted[1][4])) + curves_fitted[1][3]
      return np.abs(p_val_x - p_val)
    corr_nb_days_outside_tandem_std_comp.append(
        int(np.ceil(minimize_scalar( # Find the number of cycles
            f_1_sqrt_n_minus_value_unc_comp).x * repeat_cycle_in_days)))

  corr_nb_days_outside_tandem_std_main = np.array(corr_nb_days_outside_tandem_std_main)
  corr_nb_days_outside_tandem_std_comp = np.array(corr_nb_days_outside_tandem_std_comp)

  return corr_nb_days_outside_tandem_std_main, corr_nb_days_outside_tandem_std_comp