from typing import Optional
import numpy as np
import xarray as xr 

def meanvar(var_mean: np.ndarray, weight: np.ndarray, count: Optional[np.ndarray] = None,
            count_min: Optional[int] = 0, percent_valid_min: Optional[float] = 50,
            latmax: Optional[float] = None, latmin: Optional[float] = None,
            lat: Optional[np.ndarray] = None, return_valid_info: Optional[bool] = False):
    '''Computes the weighted mean of any grid
    Example: Used to compute global mean sea level time series: apply this function on the sea level grid for each timestep
    with 'weight' considering the ocean surface ratio on the cell and the surface of the cell.
    gmsl[i0] = meanvar(sl[i0,:,:], ocean_surface_ratio_loc)

    Parameters
    ----------
    var_mean : mask array
        the gridded variable to mean (NxM)
    weight : mask array
        the grid of weights (NxM)
    count : mask array
        can be used to mask the input grid (NxM)
    count_min : int
        Threshold applied to the count array. Default value: 0 (opt)
    percent_valid_min : float
        Default value: 50 (opt)
    latmax : float
        Default value: None (opt), remove latitude superior to latmax when computing mean
    lat : array float
        Default value: None (opt), needs to be set if latmax is set. Indicate latitude of var_mean grid

    Returns
    -------
    var_global_mean : float
        value of the mean considering the weights and conditions given as input
    '''
    if np.shape(weight) != np.shape(var_mean):
        raise Exception('var/weight mismatch')
    if count_min > 0:
        if count is None:
            raise Exception('if count_min>0, count variable must contain count grid')
        var_mean.mask[count < count_min] = True

    weight_troncated = weight.copy()
    if latmax is not None:
        if lat is None:
            raise Exception('Latitude is needed to mask out values over latmax and under -latmax')
        if latmin is not None:
            weight_troncated[lat > latmax, :] = 0.
            weight_troncated[lat < latmin, :] = 0.
        else:
            weight_troncated[np.abs(lat) > latmax, :] = 0.

    valid_info = dict()
    valid_info['n_valid'], valid_info['n_valid_max'] = np.sum(~var_mean.mask[weight_troncated > 0.]), np.sum(weight_troncated > 0.)
    assert valid_info['n_valid_max'] > 0
    valid_info['ratio_valid'] = valid_info['n_valid'] * 100. / (1. * valid_info['n_valid_max'])
    if valid_info['ratio_valid'] >= percent_valid_min:
        var_global_mean = np.ma.average(var_mean, weights=weight_troncated)
    else:
        var_global_mean = np.nan

    if return_valid_info:
        return var_global_mean, valid_info
    return var_global_mean

def ponderation_for_surface_cell(lon, lat, method='spheric'):
    '''
    Computes the surface of cells defined by lon and lat grid (MxN).

    Parameters
    ----------
    lon : array
        longitude in degrees, corresponding to cell center, size M
    lat : array
        latitude in degrees, corresponding to cell center, size N
    method : str
        Defines how the surface is computed. If "geodetic" the Earth oblateness is taken into account for the surface calculation,
        if "spheric" it is not taken into account, by default 'spheric' (opt)

    Returns
    -------
    weight: array
        ponderation grid, size NxM
    surface: array
        surface grid, size NxM

    '''

    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    assert np.all((np.diff(lat) - dlat) < 1E-5) and np.all((np.diff(lon) - dlon) < 1E-5) and dlat > 0. and dlon > 0., 'expecting regular spaced and increasing lat,lon'

    coef_bande = (4 * np.pi * (6370 * 1e3)**2) / len(lon)
    # TODO lucile: create constant file and load constant from that file
    if method == 'geodetic':
        # print('Computing surface cell with geodetic earth model')
        f = 1. / 0.29825765000000E+03  # aplatissement
        umf2 = (1. - f)**2
        ep2 = (1. - umf2) / umf2
        # ds  = longueur d'un élément de méridienne ellipsoidale
        # weight = poids par latitude géodésique
        lat_sph = np.rad2deg(np.arctan(umf2 * np.tan(np.deg2rad(lat))))
        ds = umf2 * (np.sqrt(1. + ep2 * np.sin(np.deg2rad(lat_sph))**2)**3)
        pds = ds * np.cos(np.deg2rad(lat_sph)) * (1. + f * np.sin(np.deg2rad(lat_sph))**2)
        weight = pds / sum(pds)
        sur = pds / sum(pds)
        weight *= 1. / max(weight)
    elif method == 'spheric':
        print('Computing surface cell with spheric earth model')
        weight = (np.sin(np.deg2rad(lat + dlat / 2)) - np.sin(np.deg2rad(lat - dlat / 2))) / 2
        sur = np.copy(weight)
        weight *= 1. / max(weight)
    else:
        raise IOError(f'Unknwon method {method} for computing surface cell')

    weight_total = np.matrix([sur,] * len(lon)).transpose()
    weight_total_normalized = np.matrix([weight,] * len(lon)).transpose()
    surface = coef_bande * weight_total

    return np.squeeze(np.asarray(weight_total_normalized)), np.squeeze(np.asarray(surface))

def pond_grid(dlon: Optional[int] = 3, dlat: Optional[int] = 1):
    """
    Compute a grid with ponderation weights based on water surface ratio and geodetic cell surface.

    :param dlon: Longitude resolution of the grid in degrees, default is 3°,
    :type dlon: int
    :param dlat: Latitude resolution of the grid in degrees, default is 1°,
    :type dlat: int
    :return: A tuple containing:
        - lat: Array of latitude midpoints of the grid cells,
        - cell_surface: Array of geodetic cell surface weights,
        - ponderation_grid: Array of ponderation weights for the grid,
    :rtype: tuple (numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    ds_water_ratio = xr.open_dataset('aselsu/input_data/ocean_surface_ratio_%i_%i.nc' % (dlon, dlat))
    lat_edge = ds_water_ratio.latitude_edge.values
    lon_edge = ds_water_ratio.longitude_edge.values
    lat = 0.5 * (lat_edge[0: -1] + lat_edge[1:])
    lon = 0.5 * (lon_edge[0: -1] + lon_edge[1:])

    water_ratio = np.ma.masked_invalid(ds_water_ratio.ocean_surface_ratio.values)
    weight, cell_surface = ponderation_for_surface_cell(lon, lat, method='geodetic')

    ponderation_grid = water_ratio * weight

    latmax = 66.
    ponderation_grid[np.abs(lat) > latmax, :] = 0.

    return lat, cell_surface, ponderation_grid

def weighted_standard_deviation(grid, cell_surface, weight, lat):
    """
    Calculate the weighted standard deviation.

    grid: 2D array of data values
    cell_surface: 2D array of grid cell areas
    weigh: Ponderation grid

    Returns: Weighted standard deviation
    """
    
    weighted_mean = meanvar(grid, weight, latmax=66.6, lat=lat,percent_valid_min=40.)
    variance = np.sum(cell_surface * (grid - weighted_mean)**2) / (np.sum(cell_surface)-1)
    return np.sqrt(variance)
