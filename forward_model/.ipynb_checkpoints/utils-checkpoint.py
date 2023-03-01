'''Utility functions to be used in specific calculations.'''
from math import sqrt, pow, sin, cos, radians, pi


def distance_between_points(point1_x, point1_y, point2_x, point2_y):
    '''calculate distance between two points'''
    return sqrt((point2_x - point1_x)**2 + (point2_y - point1_y)**2)


def rotate_degrees(theta):
    '''transform degrees from counter clockwise to clockwise direction'''
    return (360 - theta + 90) % 360


def rotate_radians(rad):
    '''transform radians from counter clockwise to clockwise direction'''
    return (2*pi - rad + 0.5*pi) % (2*pi)


def get_particle_diameter(phi):
    '''calculate particle diameter for given phi class'''
    return 2**(-1 * phi) * 10**-3


class InvisibleProgressBar():
    def update():
        pass
class ExecutionProgressBar():
    def __init__(self, total, desc):
        
        self.pbar = None
        try:
            from tqdm.notebook import tqdm
            self.pbar = tqdm(
                total=total, 
                colour="blue", 
                desc=desc
            )
        except Exception:
            self.pbar = InvisibleProgressBar()

    def update(self):
        self.pbar.update()