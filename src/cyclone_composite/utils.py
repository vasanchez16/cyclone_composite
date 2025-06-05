import numpy as np
import math

def get_coord_midpoints(
    latitiude_arr: np.ndarray,
    longitude_arr: np.ndarray
) -> tuple:

    lat_midpoints = np.array([ (latitiude_arr[i+1] + latitiude_arr[i])*0.5 for i in range(len(latitiude_arr)-1)])
    lon_midpoints = np.array([ (longitude_arr[i+1] + longitude_arr[i])*0.5 for i in range(len(longitude_arr)-1)])

    return lat_midpoints, lon_midpoints

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