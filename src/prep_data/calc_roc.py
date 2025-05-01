import numpy as np

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

def get_dmslp(
    mslp_arr: np.ndarray
) -> float:
    """
    Calculate the change in sea level pressure (dmslp) between two consecutive points.
    Arguments:
    mslp_arr: array-like
        Array of sea level pressure values. Units should be in hPa.
    Returns:
    dmslp: float
        Change in sea level pressure between two consecutive points.
    """


    return None