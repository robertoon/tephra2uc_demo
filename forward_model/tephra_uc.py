import json
import os
import time
import folium
import numpy as np
import multiprocessing as mp
from scipy.stats import norm
from math import exp, cos, sin, atan2, sqrt, radians
from .constants import RHO_A_S, G, DEG2RAD, AIR_VISC
from .utils import get_particle_diameter, rotate_radians, ExecutionProgressBar, polar2cartesian
from .grids import (
    generate_disk_grid, generate_ground_grid, get_custom_points
)


class TephraUC():
    def __init__(self, config):
        # config = None

        # # TODO: change config file format from json to ini
        # with open(config_file, 'r') as fp:
        #     config = json.load(fp)

        # target_path = config_file.replace(config_file.split('/')[-1], '')
        target_path = os.path.abspath(os.curdir)
        
        self.vent_easting = 10_000_000
        self.vent_northing = 10_000_000

        self.vent_location = config['vent_location']
        self.vent_height = config['vent_height']

        self.distance_from_vent_to_top_border = config["distance_from_vent_to_top_border"]
        self.distance_from_vent_to_bottom_border = config["distance_from_vent_to_bottom_border"]
        self.distance_from_vent_to_left_border = config["distance_from_vent_to_left_border"]
        self.distance_from_vent_to_right_border = config["distance_from_vent_to_right_border"]

        # self.min_easting = self.vent_easting - config['min_easting_adj']
        # self.max_easting = self.vent_easting + config['max_easting_adj']
        # self.min_northing = self.vent_northing - config['min_northing_adj']
        # self.max_northing = self.vent_northing + config['max_northing_adj']

        self.ground_grid_step = config['ground_grid_step']
        self.disk_grid_step = config['disk_grid_step']
        self.disk_radius = config["disk_radius"]
        self.total_erupted_mass = config["total_erupted_mass"]
        self.column_height = config["column_height"]
        self.umbrella_height = self.column_height + self.vent_height
        self.topo_elevation = config["topo_elevation"]
        self.col_steps = self.column_height / 500
        self.bulk_density = config["bulk_density"]
        self.atm_level_step = 500

        self.particle_density_max = config["particle_density_max"]
        self.particle_density_min = config["particle_density_min"]
        self.diffusion_coef = config["diffusion_coef"]
        self.max_phi = config["max_phi"]
        self.min_phi = config["min_phi"]
        self.tgsd_mean = config["tgsd_mean"]
        self.tgsd_sigma = config["tgsd_sigma"]
        self.simulated_max_phi = config["simulated_max_phi"]
        self.simulated_min_phi = config["simulated_min_phi"]
        self.step_phi = config["step_phi"]

        self.run_mode = config['run_mode']
        self.output_file_name = target_path + '/' + config['output_file_name']
        
        self.wind_file = None
        if config.get('wind_file'):
            self.wind_file = target_path + '/' + config['wind_file']
        else:
            self.wind_speed = config.get('wind_speed')
            self.wind_direction = config.get('wind_direction')

        if not self.wind_file and (not self.wind_direction or not self.wind_speed):
            print(f'No wind file was found! Please provide both wind speed and wind direction!')
            return

        phis = np.arange(
            self.max_phi, 
            self.min_phi + self.step_phi, 
            self.step_phi
        )
        self.phi_classes = [float(round(phi, 1)) for phi in phis]

        simulated_phis = np.arange(
            self.simulated_max_phi, 
            self.simulated_min_phi + self.step_phi, 
            self.step_phi
        )        
        self.simulated_phi_classes = [float(round(phi, 1)) for phi in simulated_phis]        

        if self.run_mode == 'grid':
            distances = {
                'from_vent_to_left': self.distance_from_vent_to_left_border,
                'from_vent_to_right': self.distance_from_vent_to_right_border,
                'from_vent_to_bottom': self.distance_from_vent_to_bottom_border,
                'from_vent_to_top': self.distance_from_vent_to_top_border
            }

            self.ground_points = generate_ground_grid(
                distances, 
                self.vent_location, 
                {'easting': self.vent_easting, 'northing': self.vent_northing},
                self.ground_grid_step
            )
            

        if self.run_mode == 'custom points':
            self.custom_points_file = target_path + '/' + config.get('custom_points_file')
            self.ground_points = get_custom_points(
                self.custom_points_file, self.vent_location, 
                (self.vent_easting, self.vent_northing)
            )
     
        print(f'Total ground points: {len(self.ground_points)}')
        
    def get_map(self):
        map = folium.Map(location=[self.vent_location[0], self.vent_location[1]], zoom_start=5)
        folium.Marker([self.vent_location[0], self.vent_location[1]]).add_to(map)

        for point in self.ground_points:
            folium.Marker(
                [point['lat'], point['lon']], 
                icon=folium.DivIcon(html=f"""<div style="color: red">x</div>"""),
                popup=f"{point['easting']}, {point['northing']}"
            ).add_to(map)
        
        return map

    def get_particle_density(self, phi):
        '''calculate particle density for given phi class'''
        max_grain_size = min(self.phi_classes)
        min_grain_size = max(self.phi_classes)

        if phi < max_grain_size:
            return self.particle_density_min
        elif phi >= max_grain_size and phi < min_grain_size:
            num = (self.particle_density_max - self.particle_density_min) * \
                (phi - min_grain_size)
            denom = max_grain_size - min_grain_size
            return self.particle_density_max - (num / denom)
        else:
            return self.particle_density_max

    def get_mass_fraction(self, phi):
        '''calculate mass fraction for given phi class'''
        p1 = norm.cdf(phi, self.tgsd_mean, self.tgsd_sigma)
        p2 = norm.cdf(phi + self.step_phi, self.tgsd_mean, self.tgsd_sigma)
        return p2 - p1

    def get_fall_time(self, phi, elevation, layer):
        """calculate fall time for a <phi> class particle falling
        from <elevation> through a given atmospheric <layer>
        """
        v_layer = 0
        rho_a = RHO_A_S * exp(-1 * elevation / 8200)
        rho = self.get_particle_density(phi) - rho_a

        v_l = (G * get_particle_diameter(phi)**2 * rho) / (18 * AIR_VISC)
        v_i = get_particle_diameter(
            phi) * ((4 * G**2 * rho**2) / (225 * AIR_VISC * rho_a))**(1/3)
        v_t = ((3.1 * rho * G * get_particle_diameter(phi))/(rho_a))**0.5

        re_l = (get_particle_diameter(phi) * rho_a * v_l) / AIR_VISC
        re_i = (get_particle_diameter(phi) * rho_a * v_i) / AIR_VISC
        re_t = (get_particle_diameter(phi) * rho_a * v_t) / AIR_VISC

        if re_l < 6:
            v_layer = v_l
        elif re_t >= 500:
            v_layer = v_t
        else:
            v_layer = v_i

        fall_time = layer/v_layer
        return fall_time

    def total_fall_time(self, phi):
        fall_time = 0
        layer = (self.umbrella_height - self.vent_height)/self.col_steps

        elevation = self.vent_height

        for i in np.arange(0, self.col_steps + 1):
            fall_time += self.get_fall_time(phi, elevation, layer)
            elevation += layer

        fall_time_adj = self.get_fall_time(
            phi, self.vent_height, self.vent_height - self.topo_elevation)
        return fall_time + fall_time_adj

    def get_particle_mass_fractions(self):
        '''return a dictionary with mass fractions for each phi class'''
        particle_mass_fractions = {}
        sum_mass_fraction = 0
        for phi in self.phi_classes:
            particle_mass_fractions[phi] = self.get_mass_fraction(phi)
            sum_mass_fraction += particle_mass_fractions[phi]

        # sum of all particle_mass_fractions should be 1
        # we divide the mass fraction error to each phi class
        mass_fraction_error = 1 - sum_mass_fraction
        if mass_fraction_error > 0:
            mass_fraction_error_unit = mass_fraction_error / \
                len(self.phi_classes)
            for phi in self.phi_classes:
                particle_mass_fractions[phi] += mass_fraction_error_unit
        # in this point, sum of all particle_mass_fractions should be 1

        return particle_mass_fractions

    def interpolate_wind_profile(self, wind_file):
        init_wind_levels = []
        final_wind_levels = []

        ht_level = self.vent_height
        col_step = (self.umbrella_height - self.vent_height)/self.col_steps

        fh = open(wind_file, 'r')
        for line in fh:
            if line.strip() == "":
                continue
            h, ws, wd = line.strip().split()
            init_wind_levels.append({
                'h': int(h),  # height
                's': float(ws),  # speed
                'd': float(wd)  # direction
            })
        fh.close()

        for _ in np.arange(0, self.col_steps + 1):
            wind = {}

            ht0 = 0.0
            dir0 = 0.0
            sp0 = 0.0

            for wind_level in init_wind_levels:
                if wind_level['h'] >= ht_level:
                    if wind_level['h'] == ht_level:
                        wind['d'] = wind_level['d']
                        wind['s'] = wind_level['s']
                    else:
                        wind['d'] = (
                            (wind_level['d'] - dir0) *
                            (ht_level - ht0) / (wind_level['h'] - ht0)
                        ) + dir0
                        wind['s'] = ((wind_level['s'] - sp0) * (ht_level - ht0) /
                                     (wind_level['h'] - ht0)) + sp0

                    wind['h'] = ht_level
                else:
                    ht0 = wind_level['h']
                    dir0 = wind_level['d']
                    sp0 = wind_level['s']

                if 'h' not in wind:
                    wind['h'] = ht_level
                    wind['s'] = sp0
                    wind['d'] = dir0

            wind['d'] *= DEG2RAD
            ht_level += col_step
            final_wind_levels.append(wind)

        return final_wind_levels

    def get_average_wind(self, phi, wind_levels):
        col_step = (self.umbrella_height - self.vent_height)/self.col_steps

        wind_speed = (wind_levels[0]['s'] * self.topo_elevation) / self.vent_height
        cos_wind = cos(wind_levels[0]['d']) * wind_speed
        sin_wind = sin(wind_levels[0]['d']) * wind_speed

        wind_x = 0
        wind_y = 0

        phi_fall_time = 0
        for idx, wind in enumerate(wind_levels):
            part_fall_time = self.get_fall_time(phi, wind['h'], col_step)
            phi_fall_time += part_fall_time
            wind_x += part_fall_time * wind['s'] * cos(wind['d'])
            wind_y += part_fall_time * wind['s'] * sin(wind['d'])

        fall_time_adj = self.get_fall_time(
            phi, self.vent_height, self.vent_height - self.topo_elevation
        )
        phi_fall_time += fall_time_adj

        x_adj = cos_wind * fall_time_adj
        y_adj = sin_wind * fall_time_adj
        average_wind_speed_x = (wind_x + x_adj)/phi_fall_time
        average_wind_speed_y = (wind_y + y_adj)/phi_fall_time

        if not average_wind_speed_x:
            average_wind_speed_x = 0.001

        if not average_wind_speed_y:
            average_wind_speed_y = 0.001

        average_wind_direction = atan2(
            average_wind_speed_y, average_wind_speed_x
        )

        # Find the average wind speed ( magnitude of the velocity vector)
        average_wind_speed = sqrt(
            average_wind_speed_x * average_wind_speed_x +
            average_wind_speed_y * average_wind_speed_y
        )

        return (average_wind_speed, average_wind_direction)

    def get_wind_components(self):
        wind_components = {}
                
        if self.wind_file:
            wind_levels = self.interpolate_wind_profile(self.wind_file)
            
            for phi in self.simulated_phi_classes:
                wind_speed, wind_direction = self.get_average_wind(
                    phi, wind_levels
                )
                # TODO: add explanation here for rotate_radians usage
                u_wind = wind_speed * cos(rotate_radians(wind_direction))
                v_wind = wind_speed * sin(rotate_radians(wind_direction))
             
                wind_components[phi] = (u_wind, v_wind)
        else:
            u_wind = self.wind_speed * cos(radians(polar2cartesian(self.wind_direction)))
            v_wind = self.wind_speed * sin(radians(polar2cartesian(self.wind_direction)))

            wind_components = {
                phi: (u_wind, v_wind) for phi in self.simulated_phi_classes
            }

       
        return wind_components

    def get_mass_accumulation(self):
        start = time.time()

        disk_grid = generate_disk_grid(
            self.vent_easting, self.vent_northing, 
            self.disk_grid_step, self.disk_radius
        )

        particle_mass_fractions = self.get_particle_mass_fractions()

        particle_fall_times = {
            phi: self.total_fall_time(phi)
            for phi in self.simulated_phi_classes
        }

        particle_mass_in_disk_cell = {
            phi: (self.total_erupted_mass / len(disk_grid)) * particle_mass_fractions[phi]            
            for phi in self.simulated_phi_classes
        }

        wind_components = self.get_wind_components()

        particle_props = {
            phi: {
                'fall_time': particle_fall_times[phi],
                'mass': particle_mass_in_disk_cell[phi],
                'u_wind': wind_components[phi][0],
                'v_wind': wind_components[phi][1],
                'source_term': (1/particle_fall_times[phi]) * (particle_mass_in_disk_cell[phi]/(4 * np.pi * self.diffusion_coef)),
                'denom': 4 * self.diffusion_coef * particle_fall_times[phi]
            } 
            for phi in self.simulated_phi_classes
        }

        global get_load, get_load_at_point
        def get_load(ground_point, disk_point_x, disk_point_y, props):           
            advection = (ground_point['easting'] - (disk_point_x + props['u_wind'] * props['fall_time']))**2 / props['denom']
            diffusion = (ground_point['northing'] - (disk_point_y + props['v_wind'] * props['fall_time']))**2 / props['denom']

            return props['source_term'] * np.exp(-advection - diffusion)

        def get_load_at_point(ground_point):            
            mass_accumulator = [
                get_load(
                    ground_point, disk_grid[:,0], disk_grid[:,1], particle_props[phi]
                ) 
                for phi in self.simulated_phi_classes
            ]

            total_load_at_point = 0
            load_by_phi = {}

            for i in range(len(self.simulated_phi_classes)):
                phi_load = np.sum(mass_accumulator[i]) 
                total_load_at_point += phi_load
                load_by_phi[self.simulated_phi_classes[i]] = phi_load
            
            return {
                'ground_point': ground_point, 
                'total_load_at_point': total_load_at_point,
                'load_by_phi': load_by_phi
            }
       
        total_tephra_load = 0
        
        results = []
        execution_progress_bar = ExecutionProgressBar(
            total=len(self.ground_points), 
            desc='mass accumulation progress'
        )

        cpus = mp.cpu_count()
        pool = mp.Pool(cpus)
        
        runs = pool.imap(
            get_load_at_point, 
            [ground_point for ground_point in self.ground_points],
            chunksize=1
        )

        for result in runs:
            results.append(result)
            total_tephra_load += result['total_load_at_point']
            execution_progress_bar.update()

        pool.close()  
        pool.join()  
        
        self.write_output(results)

        # simulation messages
        print(f'Total mass accumulation: {total_tephra_load}')
        print('Simulated phi classes: {}'.format(str(self.simulated_phi_classes)))
        print('Wind file: {}'.format(self.wind_file))
        print('Total erupted mass: {} kg'.format(self.total_erupted_mass))
        print('Disk radius: {} m'.format(self.disk_radius))
        print('Umbrella height: {} m.a.s.l.'.format(self.umbrella_height))
        print(f'Disk grid resolution: {self.disk_grid_step}x{self.disk_grid_step} m')
        print(f'Ground grid resolution: {self.ground_grid_step}x{self.ground_grid_step} m')
        print('Execution time: {:.2f} s'.format(time.time() - start))
        print('Results are stored in: {}'.format(self.output_file_name))

    def write_output(self, results):
        out = open(self.output_file_name, 'w')
        phis_columns = ','.join([f'phi_{phi}' for phi in self.simulated_phi_classes])
        out.write(f'easting,northing,lat,lon,tephra_load,{phis_columns}\n')        
                
        for result in results:          
            result_line = (
                f"{result['ground_point']['easting']:.0f},"
                f"{result['ground_point']['northing']:.0f},"
                f"{result['ground_point']['lat']:.16f},"
                f"{result['ground_point']['lon']:.16f},"
                f"{result['total_load_at_point']:.6f},"
            )

            result_line += ','.join([f"{result['load_by_phi'][phi]:.6f}" for phi in result['load_by_phi']])            
            result_line += '\n'    
           
            out.write(result_line)
        out.close()
