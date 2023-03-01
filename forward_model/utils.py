'''Utility functions to be used in specific calculations.'''
from faulthandler import disable
import numpy as np
from geopy.distance import geodesic


def polar2cartesian(theta):
    '''convert polar to cartesian wind field data'''
    return (360 - theta + 90) % 360


def distance_between_points(point1_x, point1_y, point2_x, point2_y):
    '''calculate distance between two points'''
    return np.sqrt((point2_x - point1_x)**2 + (point2_y - point1_y)**2)


def rotate_degrees(theta):
    '''transform degrees from counter clockwise to clockwise direction'''
    return (360 - theta + 90) % 360


def rotate_radians(rad):
    '''transform radians from counter clockwise to clockwise direction'''
    return (2 * np.pi - rad + 0.5 * np.pi) % (2 * np.pi)


def get_particle_diameter(phi):
    '''calculate particle diameter for given phi class'''
    return 2 ** (-1 * phi) * 10 ** -3


def polar2cartesian(theta):
    '''convert polar to cartesian wind field data'''
    return (360 - theta + 90) % 360


def lat_lon_to_3d(lat_r, lon_r):
    """
    Convert a point given latitude and longitude in radians to
    3-dimensional space, assuming a sphere radius of one.
    """
    return np.array((
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r)
    ))


def angle_between_vectors_degrees(u, v):
    """
    Return the angle between two vectors in any dimension space,
    in degrees.
    """
    return np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))


def custom_point_lat_long_to_map_utm(custom_point, vent_location, vent_utm):
    """
    Return the map utm position of a (lat long) point relative to vent false utm 
    1) get distance between points, in meters
    2) get angle between 3 points (third point being the projection of custom point on the projection of vent)
    3) get easting and northing using right triangle properties
    """

    distance = geodesic(custom_point, vent_location).meters 
    pr = (custom_point[0], vent_location[1])

    a = np.radians(np.array(vent_location))
    b = np.radians(np.array(custom_point))
    c = np.radians(np.array(pr))

    # The points in 3D space
    a3 = lat_lon_to_3d(*a)
    b3 = lat_lon_to_3d(*b)
    c3 = lat_lon_to_3d(*c)

    # Vectors in 3D space
    a3vec = a3 - b3
    c3vec = c3 - b3

    # Find the angle between the vectors
    theta = angle_between_vectors_degrees(a3vec, c3vec)
    
    northing_sign = -1 if custom_point[0] < vent_location[0] else 1
    northing = vent_utm[0] + northing_sign * distance * np.sin(np.radians(theta))

    easting_sign = -1 if custom_point[1] < vent_location[1] else 1
    easting = vent_utm[1] + easting_sign * distance * np.cos(np.radians(theta))
    
    return int(easting), int(northing)

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

class InvisibleProgressBar():
    def update():
        pass
class ExecutionProgressBar():
    def __init__(self, total, desc):
        
        self.pbar = None
        
        try:
            if is_notebook():
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm

            self.pbar = tqdm(
                total=total, 
                colour="blue", 
                desc=desc,
                leave=False
            )
        except Exception:
            self.pbar = InvisibleProgressBar()

    def update(self):
        self.pbar.update()
