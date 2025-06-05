from .locate_centers import find_candidate_points, find_centers
from .rate_of_change import get_dx, get_dy, first_order_rate_of_change, get_roc_on_full_grid
from .utils import get_coord_midpoints, haversine

import numpy as np
from global_land_mask import globe

def get_centers(
    press_data,
    latitude_arr,
    longitude_arr,
    first_order_max = 4E-5,
    second_order_min = 9E-5,
    pressure_max = 1015
):

    dx = get_dx(latitude_arr, longitude_arr)
    dy = get_dy(latitude_arr)

    lat_midpoints, lon_midpoints = get_coord_midpoints(latitude_arr, longitude_arr)

    x_1roc, y_1roc, x_2roc, y_2roc = get_roc_on_full_grid(latitude_arr, longitude_arr, lat_midpoints, lon_midpoints, dx, dy, press_data)

    limits = (first_order_max, second_order_min)
    center_candidate_points = find_candidate_points(x_1roc, y_1roc, x_2roc, y_2roc, limits)

    lons_grid, lats_grid = np.meshgrid(longitude_arr, latitude_arr)
    lats_grid = lats_grid.flatten()
    lons_grid = lons_grid.flatten()

    cyclone_centers_flat = center_candidate_points.flatten()

    candidate_points_lats = lats_grid[cyclone_centers_flat]
    candidate_points_lons = lons_grid[cyclone_centers_flat]

    centers = []

    for lat, lon in zip(candidate_points_lats, candidate_points_lons):
        temp_msl = press_data.copy()

        over_land = globe.is_land(lat, lon)

        dists = haversine((lat, lon), (lats_grid, lons_grid))
        dists = np.reshape(dists, (latitude_arr.shape[0], longitude_arr.shape[0]))
        bool_arr = (temp_msl < pressure_max) & (dists < 2000) & (center_candidate_points) & (~over_land)
        temp_msl[~bool_arr] = np.nan

        centers.append(temp_msl == np.nanmin(temp_msl))
    centers = np.array(centers)    
    centers = np.any(centers, axis=0)

    return centers, lats_grid, lons_grid