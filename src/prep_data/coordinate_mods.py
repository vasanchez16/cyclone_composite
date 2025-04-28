import numpy as np

def get_coord_midpoints(
    latitiude_arr: np.ndarray,
    longitude_arr: np.ndarray
) -> tuple:

    lat_midpoints = np.array([ (latitiude_arr[i+1] + latitiude_arr[i])*0.5 for i in range(len(latitiude_arr)-1)])
    lon_midpoints = np.array([ (longitude_arr[i+1] + longitude_arr[i])*0.5 for i in range(len(longitude_arr)-1)])

    return lat_midpoints, lon_midpoints