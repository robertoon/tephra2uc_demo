import numpy as np
from math import sqrt
from .utils import distance_between_points, custom_point_lat_long_to_map_utm


def generate_disk_grid(center_x, center_y, grid_step, disk_radius):
    '''
    Defining the umbrella cloud:
    1) generate equidistant points in the square enclosing the disk using a grid step
    2) keep only the points enclosed by disk radius
    '''
    values = np.arange(grid_step, 2 * disk_radius + grid_step, grid_step)
    
    x_term = int(center_x - disk_radius - grid_step/2)
    y_term = int(center_y - disk_radius - grid_step/2)

    meshgrid = np.meshgrid(values + x_term, values + y_term, indexing='ij')
    cartesian_prod = np.stack(meshgrid, axis=-1).reshape(-1,2)
    
    row_mask = (
        distance_between_points(
            cartesian_prod[:,0], cartesian_prod[:,1], center_x, center_y
        ) <= disk_radius
    )

    return cartesian_prod[row_mask,:]        


def lat_long_offset(lat, lon, x, y):   
    # Earth’s radius in meters
    earth_radius = 6378137

    # coordinate offsets in radians
    delta_lat = x/earth_radius
    delta_lon = y/(earth_radius * np.cos(np.pi * lat/180))

    # OffsetPosition, decimal degrees
    lat_offset = lat + delta_lat * 180/np.pi
    lon_offset = lon + delta_lon * 180/np.pi
    
    return lat_offset, lon_offset


def generate_ground_grid(distances, vent_location, vent_utm, grid_step):
    '''generate the ground grid on which tephra load is calculated'''
    ground_grid = []
    
    map_length = distances['from_vent_to_left'] + distances['from_vent_to_right']
    map_width = distances['from_vent_to_bottom'] + distances['from_vent_to_top']

    for i in range(0, map_length + grid_step, grid_step):    
        for j in range(0, map_width + grid_step, grid_step):
            point_lat_lon = lat_long_offset(
                vent_location[0], 
                vent_location[1], 
                distances['from_vent_to_top'] - j, 
                i - distances['from_vent_to_left']
            )
        
            if i <= distances['from_vent_to_left']:
                point_easting = vent_utm['easting'] - distances['from_vent_to_left'] + i
            else:
                point_easting = vent_utm['easting'] + i - distances['from_vent_to_left']
            
            if j <= distances['from_vent_to_top']:
                point_northing = vent_utm['northing'] + distances['from_vent_to_top'] - j
            else:
                point_northing = vent_utm['northing'] - (j - distances['from_vent_to_top'])
        
            ground_grid.append({
                'lat': point_lat_lon[0],
                'lon': point_lat_lon[1],
                'easting': point_easting,
                'northing': point_northing
            })
    
    return ground_grid


def get_custom_points(custom_points_file, vent_location, vent_utm):
    '''
    Convert (lat, lon) points from input file to relative map utm values
    '''
    custom_points = []
    lats, lons = np.genfromtxt(custom_points_file, skip_header=1, delimiter=',', unpack=True)

    for idx, lat in enumerate(lats):
        lon = lons[idx]
        easting, northing = custom_point_lat_long_to_map_utm(
            (lat, lon), vent_location, vent_utm
        )
        point = {
            'lat': lat,
            'lon': lon,
            'easting': easting,
            'northing': northing
        }
        custom_points.append(point)

    return custom_points
