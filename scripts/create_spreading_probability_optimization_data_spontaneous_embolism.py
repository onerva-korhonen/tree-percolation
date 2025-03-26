"""
A script for creating data for optimizing SI spreading probability in the presence of spontaneous embolism (this means simulating 
embolism spreading for a set of pressure differences and spreading probabilities;
the evolution of effective conductance and network properties are saved for each pressure difference / spreading probability).

This script creates the data needed for Figs. 2-4 of the manuscripti.

After this, run optimize_spreading_probability.py to find optimal spreading probability for each pressure difference.

Written by Onerva Korhonen
"""
import mrad_model
import mrad_params
import params
import percolation
import simulations

import numpy as np
import openpnm as op
import sys
import pickle

# general parameters; these are common for all runs
# NOTE: do not change these parameters here; to modify parameters, change params.py

cfg = {}
cfg['Dc'] = params.Dc
cfg['Dc_cv'] = params.Dc_cv
cfg['conduit_element_length'] = params.Lce
cfg['fc'] = params.fc
cfg['average_pit_area'] = params.average_pit_membrane_area
cfg['fpf'] = params.fpf
cfg['tf'] = params.tf
cfg['Dp'] = params.truncnorm_center
cfg['Tm'] = params.Tm
cfg['n_constrictions'] = params.n_constrictions
cfg['truncnorm_center'] = params.truncnorm_center
cfg['truncnorm_std'] = params.truncnorm_std
cfg['truncnorm_a'] = params.truncnorm_a
cfg['pore_shape_correction'] = params.pore_shape_correction
cfg['gas_contact_angle'] = params.gas_contact_angle
cfg['icc_length'] = params.icc_length
cfg['si_length'] = params.si_length
cfg['si_tolerance_length'] = params.si_tolerance_length
cfg['start_conduits'] = params.start_conduits
cfg['surface_tension'] = params.surface_tension
cfg['nCPUs'] = params.nCPUs
cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path

cfg['NPc'] = params.NPc
cfg['Pc'] = params.Pc
cfg['Pe_rad'] = params.Pe_rad
cfg['Pe_tan'] = params.Pe_tan

# parameters specific to this run
cfg['net_size'] = [100, 10, 100]
cfg['bpp_type'] = 'young-laplace_with_constrictions'
cfg['spontaneous_embolism'] = True

vulnerability_pressure_range = np.arange(0, 8, step=0.25)*1E6
small_spreading_probability_range = np.arange(0.01, 0.15, step=0.005)#np.logspace(np.log10(0.0001), np.log10(0.02), 15)
large_spreading_probability_range = np.arange(0.01, 0.05, step=0.01)
# using two probability ranges is a hack for combining data calculated in different spaces

include_orig_values = True

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

percolation_type = 'si'
removal_order = 'random'
break_nonfunctional_components = False

create_networks = False

#n_iterations = 100
small_pressures = [[pressure] for pressure in vulnerability_pressure_range] + [[] for i in range(len(small_spreading_probability_range))]
large_pressures = [[] for i in range(len(large_spreading_probability_range))]
small_probabilities = [[] for i in range(len(vulnerability_pressure_range))] + [[probability] for probability in small_spreading_probability_range]
large_probabilities = [[probability] for probability in large_spreading_probability_range]
n_small_pressures = len(small_pressures)
n_large_pressures = len(large_pressures)
small_large_limit = 10000 # for using only small_spreading_probability_range, this should be larger than the number of jobs

# NOTE: do not modify any parameters below this point

if __name__=='__main__':
    index = int(sys.argv[1])
    if index <= small_large_limit:
        pressures = small_pressures
        probabilities = small_probabilities
        n_pressures = n_small_pressures

        iteration_index = int(np.floor(index / n_pressures))
        pressure_index = index - n_pressures * iteration_index
    else:
        pressures = large_pressures
        probabilities = large_probabilities
        n_pressures = n_large_pressures

        iteration_index = int(np.floor((index - small_large_limit - 1) / n_pressures))
        pressure_index = (index - small_large_limit - 1) - n_pressures * iteration_index

    pressure = pressures[pressure_index]
    probability = probabilities[pressure_index]
    save_path = params.optimized_spreading_probability_save_path_base + '_' + str(index) + '.pkl'

    if create_networks:

        cfg['conduit_diameters'] = 'lognormal'

        # creating the xylem network following Mrad and preparing it for simulations with OpenPN
        conduit_elements, conns = mrad_model.create_mrad_network(cfg)
        net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
        net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
        mrad_model.prepare_simulation_network(net_cleaned, cfg, update_coords=True)

    else:
        network_save_path = params.spreading_probability_optimization_network_save_path_base + '_' + str(iteration_index) + '.pkl' 
        with open(network_save_path, 'rb') as f:
            network_data = pickle.load(f)
            f.close()
        net = network_data['network']
        start_conduits = network_data['start_conduits_random_per_component']
        cfg['start_conduits'] = start_conduits

    cfg['conduit_diameters'] = 'inherit_from_net'

    spontaneous_embolism_probabilities = percolation.get_spontaneous_embolism_probability(vulnerability_pressure_range)
    cfg['spontaneous_embolism_probabilities'] = spontaneous_embolism_probabilities


    percolation.run_spreading_iteration(net, cfg, pressure, save_path, 
                                        spreading_probability_range=probability, 
                                        si_length=cfg['si_length'], include_orig_values=True)

