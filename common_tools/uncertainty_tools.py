import lenapy as ln
import numpy as np
import xarray as xr


def compute_trend(time_vec,data,cov):
    '''Computes the trend and trend uncertainty using extended OLS. 
    Parameters
    ----------
    time_vec : numpy array(datetime64)
        Dates for each datapoint, shape (N,)
    data : numpy array(float)
        Data to fit, shape (N,)
    cov : numpy array(float)
        Covariance matrix used by the extended OLS, shape (N,N)
    
    Returns
    -------
    trend : float
        Value of the trend fitted with extended OLS, given in data units/days. Returns nan if the estimation failed.
    trend_unc : float
        trend uncertainty at one sigma estimated with extended OLS, given in data units/days. Returns nan if the estimation failed.
    '''
    obj = xr.DataArray(data=data, dims=["time"], coords=dict(time=time_vec))
    try:
        est = obj.lntime.OLS(degree=2,sigma=cov,datetime_unit="d")
        trend = est.coefficients[1].values
        trend_unc = est.uncertainties[1].values
    except:
        trend = np.nan
        trend_unc = np.nan
    return trend,trend_unc

def add_manual_error(cov,sigma_array):
    '''Function updating a lenapy covariance matrix object by adding a new error covariance matriw which is manually defined. 
    The previous error covariance matrices in the lenapy object are kept.
    WARNING : there are no checks concerning the symmetry, inversibility or shape of the matrix. 
    Parameters
    ----------
    cov : lenapy.utils.covariance.covariance 
        Lenapy covariance object
    sigma_array : numpy array(float)
        Numpy array representing a new error covariance matrix. 
    
    Returns
    -------
    None
    '''
    cov.add_errors('bias',0.0,t0=cov.time.min()) # adding a matrix of zeros
    new_err = cov.errors[-1]
    new_err.sigma = sigma_array
    new_err.type='manual'
    cov.sigma=cov.sigma + new_err.sigma