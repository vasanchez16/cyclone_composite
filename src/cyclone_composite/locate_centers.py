import numpy as np
import xarray as xr
import math
from .utils import haversine

def find_candidate_points(
    x_1roc: np.ndarray,
    y_1roc: np.ndarray,
    x_2roc: np.ndarray,
    y_2roc: np.ndarray,
    limits: tuple = (0, 0),
) -> np.ndarray:
    """
    Identify candidate cyclone centers based on the first and second order rate of change of sea level pressure.
    Arguments:
    x_1roc: np.ndarray
        First order rate of change in the x-direction (longitude).
    y_1roc: np.ndarray
        First order rate of change in the y-direction (latitude).
    x_2roc: np.ndarray
        Second order rate of change in the x-direction (longitude).
    y_2roc: np.ndarray
        Second order rate of change in the y-direction (latitude).
    limits: tuple
        Tuple containing the maximum and minimum limits for the first and second order rate of change.
    Returns:
    candidate_points: np.ndarray
        Boolean array indicating the locations of candidate cyclone centers.
    """
    # limits from Field et al. (2007)
    # max_1roc = 4E-5 # hPa km^-2
    # min_2roc = 9E-5 # hPa km^-2

    max_1roc, min_2roc = limits[0], limits[1]

    roc1_prod = x_1roc * y_1roc
    roc1_bools = roc1_prod < max_1roc

    roc2_sum = x_2roc + y_2roc
    roc2_bools = roc2_sum > min_2roc

    candidate_points = roc1_bools * roc2_bools

    return candidate_points

def find_centers(
    candidate_points: np.ndarray,
    latitude_arr: np.ndarray,
    longitude_arr: np.ndarray,
    mslp_arr: np.ndarray,
    pressure_max: float = 1015,
    distance_max: float = 2000
) -> np.ndarray:
    """
    Find the cyclone centers by identifying the maximum negative pressure anonmalies satisfying the conditions in Field et al. 2007.
    Arguments:
    candidate_points: np.ndarray
        Boolean array indicating the locations of candidate cyclone centers.
    latitude_arr: np.ndarray
        Array of latitude values.
    longitude_arr: np.ndarray
        Array of longitude values.
    mslp_arr: np.ndarray
        Array of sea level pressure values. Units should be in hPa.
    pressure_max: float
        Maximum sea level pressure value to consider for cyclone centers. Units should be in hPa.
    distance_max: float
        Maximum distance in kilometers to consider for cyclone centers.
    Returns:
    centers: np.ndarray
        Boolean array indicating the locations of cyclone centers.
    """

    lons_grid, lats_grid = np.meshgrid(longitude_arr, latitude_arr)
    lats_grid = lats_grid.flatten()
    lons_grid = lons_grid.flatten()

    candidate_points_lats = lats_grid[candidate_points.flatten()]
    candidate_points_lons = lons_grid[candidate_points.flatten()]


    centers = []
    for lat, lon in zip(candidate_points_lats, candidate_points_lons):
        temp_msl = mslp_arr.copy()

        dists = haversine((lat, lon), (lats_grid, lons_grid))
        dists = np.reshape(dists, (latitude_arr.shape[0], longitude_arr.shape[0]))
        bool_arr = (temp_msl < pressure_max) & (dists < distance_max) & (candidate_points)
        temp_msl[~bool_arr] = np.nan

        centers.append(temp_msl == np.nanmin(temp_msl))
    centers = np.array(centers)    
    centers = np.any(centers, axis=0)

    return centers