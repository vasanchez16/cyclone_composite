import numpy as np
import xarray as xr

########################################################################################################################
############################################ Rate of Change Functioins #################################################
########################################################################################################################

def get_dx(
    latitiude_arr: np.ndarray, longitude_arr: np.ndarray
):
    """
    Calculate the distance between two consecutive longitude points in km.
    Arguments:
    latitiude_arr: array-like
        Array of latitude values.
    longitude_arr: array-like
        Array of longitude values.
    Returns:
    dx: array-like
        Distance between two consecutive longitude points in km.
    """
    dlon = [longitude_arr[i+1] - longitude_arr[i] for i in range(len(longitude_arr)-1)]
    if not all(dlon == dlon[0]):
        raise ValueError("Longitude values are not equally spaced.")
    
    dx = dlon[0] * 111.320 * np.cos(np.deg2rad(latitiude_arr)) # km

    return dx

def get_dy(
    latitude_arr: np.ndarray
) -> float:
    """
    Calculate the distance between two consecutive latitude points in km.
    Arguments:
    latitude_arr: array-like
        Array of latitude values.
    Returns:
    dy: float
        Distance between two consecutive latitude points in km.
    """

    
    dlat = [latitude_arr[i+1] - latitude_arr[i] for i in range(len(latitude_arr)-1)]
    if not all(dlat == dlat[0]):
        raise ValueError("Longitude values are not equally spaced.")

    dy = dlat[0] * 110.574 # km
    
    return dy

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