import xarray as xr 
import numpy as np 
import netCDF4
from datetime import datetime,timedelta
import time
from scipy.interpolate import griddata,interp1d
import statsmodels.api as sm

from aselsu.common_tools.time_tools import julianday_to_decimalyears_array,datetime_to_decimalyears
from aselsu.common_tools import time_tools, plot_tools

## Corrections functions
### TOPEX-A correction
def tpa_drift_correction(tpa_corr_file, jd, method='v-shape'):
    """ Compute TOPEX-A drift correction for GMSL time series
    V-shape correction from Cazenave and the WCRP global sea level budget group, ESSD, 2018

    Input
    -----
    jd: numpy array
        time array [in julian days]
    method: string ['linear'|'v-shape'(default)|'v-smoothed']
        TPA drift correction method
    Returns
    -------
    tpa_corr: numpy array
        Correction for TPA drift [in meters]
    """

    dy = julianday_to_decimalyears_array(jd)
    tpa_corr = np.zeros(jd.shape)

    if method.lower() == 'linear':
        ind = np.where(dy<1999)[0]
        tpa_corr[ind] = 0.0015*(dy[ind]-dy[ind[-1]])
    elif method.lower() == 'v-shape' or method.lower() == 'v':
        date1 = datetime_to_decimalyears(datetime(1995,7,31))
        date2 = datetime_to_decimalyears(datetime(1999,2,28))
        ind1 = np.where(dy<=date1)[0]
        ind2 = np.where(dy<=date2)[0]
        tpa_corr[ind2] = 0.003*(dy[ind2]-dy[ind2[-1]])
        tpa_corr[ind1] = tpa_corr[ind1]-0.004*(dy[ind1]-dy[ind1[-1]])
    elif 'smoothed' in method.lower():
        ds_tpa = xr.open_dataset(tpa_corr_file,decode_times=False)
        jd_tpa = ds_tpa.time.values
        tpa_corr_init = ds_tpa.tpa_corr.values
        tpa_corr = np.interp(jd, jd_tpa, tpa_corr_init, left=tpa_corr_init[0], right=tpa_corr_init[-1])
    else:
        raise Exception('Unknown method in correct_for_tpa_drift function.')

    return tpa_corr


def correct_gmsl_for_tpa_drift(tpa_corr_file, jd, gmsl, method='v-shape'):
    """ Correct GMSL time series for TOPEX-A drift

    Input
    -----
    jd: numpy array
        time array [in julian days]
    gmsl: numpy array
        GMSL time series to correct [in meters]
    method: string ['linear'|'v-shape'(default)|'v-smoothed']
        TPA drift correction method
    Returns
    -------
    gmsl_corr: numpy array
        GMSL corrected for TPA drift [in meters]
    """

    tpa_corr = tpa_drift_correction(tpa_corr_file, jd,method=method)
    gmsl_corr = gmsl-tpa_corr

    return gmsl_corr

### Correction of Jason-3 wet troposphere correction drift
def correct_for_jason3_wtc_drift(ncfile, jd, gmsl):
    """ Correct GMSL time series for Jason-3 radiometer drift

    Input
    -----
    ncfile: str
        path to j3_wtc_drift_correction_cdr_al_s3a.nc file containing Jason-3 correction
    jd: numpy array
        time array [in julian days]
    gmsl: numpy array
        GMSL time series to correct [in meters]
    Returns
    -------
    gmsl_corr: numpy array
        GMSL corrected for Jason-3 WTC drift [in meters]
    """

    ds = xr.open_dataset(ncfile,decode_times=False)
    jd_j3_corr = ds.time.values
    j3_corr = np.ma.masked_invalid(ds.j3_corr)

    j3_corr_interp = np.interp(jd, jd_j3_corr, j3_corr, left=0, right=j3_corr[-1])
    gmsl_corr = gmsl-j3_corr_interp

    return gmsl_corr

def remove_periods_signals(myperiods,time_jdays,ts):
  # Define period in days
  T = myperiods
  omega = 2 * np.pi / T

  # Design matrix for OLS: sin, cos, and constant
  X = np.column_stack([np.sin(omega * time_jdays), np.cos(omega * time_jdays), 
                       np.ones_like(time_jdays)])
  ols_model = sm.OLS(ts, X).fit()
  fitted_signal = ols_model.predict(X)

  # Remove the periodic signal
  ts -= fitted_signal

  return ts

def gmsl_trend_uncertainty(error_prescription,time_vec,data):
  cov = data.lntime.covariance_analysis()
  cov.read_yaml(error_prescription)
  covar = cov.sigma

  print(f"**Uncertainty computation for {error_prescription}**\n")
  dates=[]
  dates = np.array([datetime(1950,1,1) + timedelta(el) for el in time_vec])
  time = time_tools.datetime_to_decimalyears_array(dates)
  year_start=dates[0].year
  year_end=dates[-1].year

  dict_eval = {'redo':1,'number_periods':(year_end-year_start+1),'period_min':1,
               'number_dates':(year_end-year_start+1),'number_enveloppes':4}
  dict_interp = {'redo':1,'number_dates':(year_end-year_start+1)*10,
                 'number_periods':200}

  u = plot_tools.fractal_trend_uncertainties_interp(time,dict_eval, dict_interp,
                                        covar_mat=covar*1e6, yvar=data*1e3,
                                        least_squares_method='OLS',verbose=0)
  start_period = np.where(u['period']>3.)[0][0]
  period,gmsl_trend_unc=[],[]
  for i in range(start_period, len(u['period']) - 1):
    if u['uncertainties'][i, :].mask.all():
        break 
    period.append(u['period'][i])
    gmsl_trend_unc.append(
        u['uncertainties'][i, :][~u['uncertainties'][i, :].mask][-1] * 365.25
    )
  period = np.array(period)
  gmsl_trend_unc = np.array(gmsl_trend_unc)

  return period, gmsl_trend_unc, u