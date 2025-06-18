import xarray as xr

def add_var_attributes(ds):
    """
    Function adding clean variable attributes (units, long_name and comments) to the input dataset
    It is specific to the SSB computation.
    """
    ds['ssb_unc'].attrs = {'long_name':'SSB uncertainty','units':'m','comment':'SSB uncertainty (at 1 sigma) estimated using alpha and SWH uncertainties (without the temporal correlation between SSB values)'}
    ds['ssb_alpha_unc'].attrs = {'long_name':'SSB uncertainty component from alpha','units':'m','comment':'SSB uncertainty component (at 1 sigma) from uncertainties in alpha (without the temporal correlation between SSB values)'}
    ds['ssb_swh_unc'].attrs = {'long_name':'SSB uncertainty component from SWH','units':'m','comment':'SSB uncertainty component (at 1 sigma) from uncertainties in SWH (without the temporal correlation between SSB values)'}
    ds['ssb_trend'].attrs = {'long_name':'SSB trend','units':'mm/year','comment':'SSB trend using extended OLS'}
    ds['ssb_trend_unc'].attrs ={'long_name':'SSB trend uncertainties','units':'mm/years','comment':'SSB trend uncertainty (at 1 sigma) using extended OLS'}
    ds['ssb_trend_alpha_unc'].attrs ={'long_name':'SSB trend uncertainty component from alpha','units':'mm/years','comment':'SSB trend uncertainty (at 1 sigma) from uncertainties on alpha using extended OLS'}
    ds['ssb_trend_swh_unc'].attrs ={'long_name':'SSB trend uncertainty component from SWH','units':'mm/years','comment':'SSB trend uncertainty (at 1 sigma) from uncertainties on SWH using extended OLS'}
    ds['swh_trend'].attrs = {'long_name':'SWH trend','units':'mm/year','comment':'SWH trend using extended OLS'}
    ds['swh_trend_unc'].attrs ={'long_name':'SWH trend uncertainties','units':'mm/years','comment':'SWH trend uncertainty (at 1 sigma) using extended OLS'}