import numpy as np
import xarray as xr

def first_order_rate_of_change(
    dx,
    dy,
    mslp
):
    """
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
    Calculate the first and second order rate of change of sea level pressure (mslp) on a full grid.

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
    limits: tuple = (0, 0, 0, 0),
):
    
    # max_1roc = 4E-5 # hPa km^-2
    # min_2roc = 9E-5 # hPa km^-2

    max_1roc, min_2roc = limits[0], limits[1]

    roc1_prod = x_1roc * y_1roc
    roc1_bools = roc1_prod < max_1roc

    roc2_sum = x_2roc + y_2roc
    roc2_bools = roc2_sum > min_2roc

    cyclone_centers = roc1_bools * roc2_bools

    return cyclone_centers