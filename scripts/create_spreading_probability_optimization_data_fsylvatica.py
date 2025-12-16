"""
A script for creating data for optimizing SI spreading probability (this means simulating embolism spreading for a set of pressure differences and spreading probabilities;
the evolution of effective conductance and network properties are saved for each pressure difference / spreading probability.

This script is tailored for repeating the calculations for several stem segments of F. sylvatica. It should be easily modifiable also for segment analysis in other
species. The basic idea is to run each segment, each pressure / probability, and each iteration as a job of its own; the total number of jobs is 
n_pressures x n_segments x n_iterations.

Note that number of iterations per sap pressure / spreading probability is defined by the number of simulation networks (spreading will be run on all available networks).
Number of iterations is not set in this script.

After this, run optimize_spreading_probability.py to find optimal spreading probability for each pressure difference.

Written by Onerva Korhonen
"""
import mrad_model
import mrad_params
import params
import fsylvatica_params
import percolation
import simulations

import numpy as np
import openpnm as op
import sys
import pickle

# general parameters; these are common for all runs
# NOTE: do not change these parameters here; to modify parameters, change params.py

cfg = {}
cfg['conduit_element_length'] = fsylvatica_params.Lce
cfg['tf'] = fsylvatica_params.tf
cfg['Dp'] = fsylvatica_params.truncnorm_center
cfg['Tm'] = fsylvatica_params.Tm
cfg['n_constrictions'] = fsylvatica_params.n_constrictions
cfg['truncnorm_center'] = fsylvatica_params.truncnorm_center
cfg['truncnorm_std'] = fsylvatica_params.truncnorm_std
cfg['truncnorm_a'] = fsylvatica_params.truncnorm_a
cfg['pore_shape_correction'] = fsylvatica_params.pore_shape_correction
cfg['gas_contact_angle'] = fsylvatica_params.gas_contact_angle
cfg['icc_length'] = params.icc_length
cfg['si_length'] = params.si_length
cfg['si_tolerance_length'] = params.si_tolerance_length
cfg['start_conduits'] = params.start_conduits
cfg['surface_tension'] = params.surface_tension
cfg['nCPUs'] = params.nCPUs

cfg['NPc'] = fsylvatica_params.NPc
cfg['Pc'] = fsylvatica_params.Pc
cfg['Pe_rad'] = fsylvatica_params.Pe_rad
cfg['Pe_tan'] = fsylvatica_params.Pe_tan

# parameters specific to this run
cfg['net_size'] = [100, 10, 100]
cfg['bpp_type'] = 'young-laplace_with_constrictions'
cfg['spontaneous_embolism'] = False

vulnerability_pressure_range = np.arange(0.5, 3.0, step=0.05)*1E6
vulnerability_pressure_range = [[pressure] for pressure in vulnerability_pressure_range]
probability = [] # no SI spreading calculations at the moment

include_orig_values = True

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

percolation_type = 'si'
removal_order = 'random'
break_nonfunctional_components = False

create_networks = False

zero_index = 0 # index of the first array job

# NOTE: do not modify any parameters below this point
# Note on the indexing order: calculations are performed in the iteration -> pressure/probability order (i.e. first all iterations for the first pressure etc.)

if __name__=='__main__':

    index = int(sys.argv[1])
    index_str = str(index)
    if zero_index > 0:
        index -= zero_index
        
    n_segments = len(fsylvatica_params.Dc)
    n_pressures = len(vulnerability_pressure_range)
    iteration_index = int(np.floor(index / (n_segments * n_pressures))) # for each iteration, there are n_segments x n_pressures jobs
    segment_index = int(np.floor((index - iteration_index * n_segments * n_pressures) / n_pressures)) # for each segment, there are n_pressures jobs 
    pressure_index = int(np.floor(index - iteration_index * n_segments * n_pressures - segment_index * n_pressures))

    cfg['Dc'] = fsylvatica_params.Dc[segment_index]
    cfg['Dc_cv'] = fsylvatica_params.Dc_cv[segment_index]
    cfg['fc'] = fsylvatica_params.fc[segment_index]
    cfg['fpf'] = fsylvatica_params.fpf[segment_index]
    cfg['segment_name'] = fsylvatica_params.segment_names[segment_index]
    cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path.split('.')[0] + '_' + fsylvatica_params.segment_names[segment_index] + '.pkl'
    
    pressure = vulnerability_pressure_range[pressure_index]
    save_path = params.optimized_spreading_probability_save_path_base + '_' + fsylvatica_params.segment_names[segment_index] + '_' + index_str + '.pkl'

    network_save_path = params.spreading_probability_optimization_network_save_path_base + '_' + fsylvatica_params.segment_names[segment_index] + '_' + str(iteration_index) + '.pkl' 
    with open(network_save_path, 'rb') as f:
        network_data = pickle.load(f)
        f.close()
    net = network_data['network']
    start_conduits = network_data['start_conduits_random_per_component']
    cfg['start_conduits'] = start_conduits

    cfg['conduit_diameters'] = 'inherit_from_net'

    percolation.run_spreading_iteration(net, cfg, pressure, save_path, 
                                        spreading_probability_range=probability, 
                                        si_length=cfg['si_length'], include_orig_values=True)

