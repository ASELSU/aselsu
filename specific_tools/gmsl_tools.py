import xarray as xr 
import numpy as np 
import netCDF4
from datetime import datetime,timedelta
import time
from scipy.interpolate import griddata,interp1d
from aselsu.common_tools.time_tools import julianday_to_decimalyears_array,datetime_to_decimalyears

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
