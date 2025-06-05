import numpy as np
import xarray as xr
import math

def first_order_rate_of_change(
    dx,
    dy,
    mslp
):
    """
    Calculate the first order rate of change of sea level pressure (mslp) in the x (longitude) and y (latitude) directions.
    Arguments:
    dx: np.ndarray
        Distance between two consecutive longitude points in km.
    dy: float
        Distance between two consecutive latitude points in km.
    mslp: np.ndarray
        Array of sea level pressure values. Units should be in hPa.
    Returns:
    dmslp_x_dx: np.ndarray
        First order rate of change in the x-direction (longitude).
    dmslp_y_dy: np.ndarray
        First order rate of change in the y-direction (latitude).
    """

    dmslp_x = np.diff(mslp, axis = 1)
    dmslp_y = np.diff(mslp, axis = 0)

    dmslp_x_dx = dmslp_x / dx[:, np.newaxis]
    dmslp_y_dy = dmslp_y / dy

    return dmslp_x_dx, dmslp_y_dy


def second_order_rate_of_change(
    dx,
    dy,
    dmslp_x_dx,
    dmslp_y_dy
):
    """
    Calculate the second order rate of change of sea level pressure (mslp) in the x (longitude) and y (latitude) directions.
    Arguments:
    dx: np.ndarray
        Distance between two consecutive longitude points in km.
    dy: float
        Distance between two consecutive latitude points in km.
    dmslp_x_dx: np.ndarray
        First order rate of change in the x-direction (longitude).
    dmslp_y_dy: np.ndarray
        First order rate of change in the y-direction (latitude).
    Returns:
    d2mslp_x_dx2: np.ndarray
        Second order rate of change in the x-direction (longitude).
    d2mslp_y_dy2: np.ndarray
        Second order rate of change in the y-direction (latitude).
    """

    d2mslp_x_dx = np.diff(dmslp_x_dx, axis = 1)
    d2mslp_y_dy = np.diff(dmslp_y_dy, axis = 0)

    d2mslp_x_dx2 = d2mslp_x_dx / dx[:, np.newaxis]
    d2mslp_y_dy2 = d2mslp_y_dy / dy

    return d2mslp_x_dx2, d2mslp_y_dy2

def get_roc_on_full_grid(
    latitude_arr: np.ndarray,
    longitude_arr: np.ndarray,
    latitude_midpoints: np.ndarray,
    longitude_midpoints: np.ndarray,
    dx: np.ndarray,
    dy: float,
    mslp_arr: np.ndarray
) -> tuple:
    
    """
    Interpolate the first and second order rate of change of sea level pressure (mslp) on a full grid.

    Returns:
    x_1roc: np.ndarray
        First order rate of change in the x-direction (longitude).
    y_1roc: np.ndarray
        First order rate of change in the y-direction (latitude).
    x_2roc: np.ndarray
        Second order rate of change in the x-direction (longitude).
    y_2roc: np.ndarray
        Second order rate of change in the y-direction (latitude).    
    """

    interp_method = 'linear'
    
    dmslp_x_dx, dmslp_y_dy = first_order_rate_of_change(dx, dy, mslp_arr)

    d2mslp_x_dx2, d2mslp_y_dy2 = second_order_rate_of_change(dx, dy, dmslp_x_dx, dmslp_y_dy)

    x_1roc = xr.DataArray(
        data = dmslp_x_dx,
        dims = ("latitude","longitude"),
        coords = {
            "latitude":latitude_arr,
            "longitude":longitude_midpoints
        }
    )
    x_1roc_interp = x_1roc.interp(
        latitude=latitude_arr,
        longitude=longitude_arr,
        method = interp_method
    )

    y_1roc = xr.DataArray(
        data = dmslp_y_dy,
        dims = ("latitude","longitude"),
        coords = {
            "latitude":latitude_midpoints,
            "longitude":longitude_arr
        }
    )
    y_1roc_interp = y_1roc.interp(
        latitude=latitude_arr,
        longitude=longitude_arr,
        method = interp_method
    )

    x_2roc = xr.DataArray(
        data = d2mslp_x_dx2,
        dims = ("latitude","longitude"),
        coords = {
            "latitude":latitude_arr,
            "longitude":longitude_arr[1:-1]
        }
    )
    x_2roc_interp = x_2roc.interp(
        latitude=latitude_arr,
        longitude=longitude_arr,
        method = interp_method
    )

    y_2roc = xr.DataArray(
        data = d2mslp_y_dy2,
        dims = ("latitude","longitude"),
        coords = {
            "latitude":latitude_arr[1:-1],
            "longitude":longitude_arr
        }
    )
    y_2roc_interp = y_2roc.interp(
        latitude=latitude_arr,
        longitude=longitude_arr,
        method = interp_method
    )
    


    return x_1roc_interp.data, y_1roc_interp.data, x_2roc_interp.data, y_2roc_interp.data

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

def haversine(
        coord1: tuple,
        coord2: tuple
) -> float:
    """
    Calculate the distance between two global coordinates using the Haversine formula.
    Arguments:
    coord1: tuple
        Tuple containing the latitude and longitude of the first point (lat1, lon1).
    coord2: tuple
        Tuple containing the latitude and longitude of the second point (lat2, lon2).
    Returns:
    distance: float
        Distance between the two points in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1 = lat1 * (math.pi / 180)
    lon1 = lon1 * (math.pi / 180)
    lat2 = lat2 * (math.pi / 180)
    lon2 = lon2 * (math.pi / 180)
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Calculate the distance
    distance = R * c  # Distance in kilometers
    
    return distance

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