from math import exp, pi


class TephraLoad():
    def __init__(self, disk_grid, phi_classes, particle_fall_times, particle_mass_in_disk_cell, wind_components, diffusion_coef):        
        self.disk_grid = disk_grid
        self.phi_classes = phi_classes
        self.particle_fall_times = particle_fall_times
        self.particle_mass_in_disk_cell = particle_mass_in_disk_cell
        self.wind_components = wind_components
        self.diffusion_coef = diffusion_coef
        
    def get_load(self, ground_point, disk_cell_point, part_mass, part_fall_time, u_wind, v_wind):
        '''calculate tephra accumulation at given (x, y) location
        from a given umbrella cell'''
        source_term = (1/part_fall_time) * (part_mass/(4 * pi * self.diffusion_coef))

        denom = 4 * self.diffusion_coef * part_fall_time
        advection = (ground_point['x'] - (disk_cell_point['x'] + u_wind * part_fall_time))**2 / denom
        diffusion = (ground_point['y'] - (disk_cell_point['y'] + v_wind * part_fall_time))**2 / denom

        return source_term * exp(-advection - diffusion)

    def get_load_at_point(self, ground_point):
        '''calculate tephra accumulation at given point from all umbrella cells'''
        total_load = 0

        for phi in self.phi_classes:
            particle_fall_time = self.particle_fall_times[phi]
            particle_mass = self.particle_mass_in_disk_cell[phi]
            U_WIND, V_WIND = self.wind_components[phi]

            for disk_grid_point in self.disk_grid:
                total_load += self.get_load(
                    ground_point, disk_grid_point,
                    particle_mass, particle_fall_time,
                    U_WIND, V_WIND
                )

        return {
            'ground_point': ground_point, 
            'total_load': total_load
        }
