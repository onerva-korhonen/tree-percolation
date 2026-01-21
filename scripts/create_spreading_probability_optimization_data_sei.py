"""
A script for creating data for optimizing the S-I transition probability (i.e. spreading probability)
in the SEI model of embolism spreading (the E-I transition probability, i.e. bubble expansion probability
is defined as a function of xylem pressure). The spreading probability is optimized by 
simulating  embolism spreading for a set of pressure differences (and corresponding bubble expansion probabilities) and spreading probabilities;
the evolution of effective conductance and network properties are saved for each pressure difference / spreading probability).

This script is meant for simulating embolism spreading with the SEI model and includes calculation of the bubble expansion probability.´

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
cfg['spontaneous_embolism'] = False
cfg['bubble_expansion'] = True

vulnerability_pressure_range = np.arange(0, 3.0, step=0.05)*1E6
spreading_probability_range = np.arange(0, 0.2, step=0.005)

include_orig_values = True

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

percolation_type = 'si'
removal_order = 'random'
break_nonfunctional_components = False

create_networks = False
project_specific_networks = False # set to True for using networks created for the current project, False to re-used networks created for an earlier project

project_specific_bpp = False
if not project_specific_bpp:
    cfg['bpp_data_path'] = params.alternative_bubble_propagation_pressure_data_path # set to True for using BPP data created for the current project, False to re-used BPP data created for an earlier project

#n_iterations = 100
pressures = [[pressure] for pressure in vulnerability_pressure_range] + [[] for i in range(len(spreading_probability_range))]
probabilities = [[] for i in range(len(vulnerability_pressure_range))] + [[probability] for probability in spreading_probability_range]
n_pressures = len(pressures)
zero_index = 0 # index of the first array job

# NOTE: do not modify any parameters below this point
# Note on the indexing order: calculations are performed in the iteration -> pressure/probability order (i.e. first all iterations for the first pressure etc.)

if __name__=='__main__':

    index = int(sys.argv[1])
    if zero_index > 0:
        index -= zero_index

    iteration_index = int(np.floor(index / n_pressures))
    pressure_index = index - n_pressures * iteration_index

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
        if project_specific_networks:
            network_save_path = params.spreading_probability_optimization_network_save_path_base + '_' + str(iteration_index) + '.pkl' 
        else:
            network_save_path = params.alternative_network_save_path_base + '_' + str(iteration_index) + '.pkl'

        with open(network_save_path, 'rb') as f:
            network_data = pickle.load(f)
            f.close()
        net = network_data['network']
        start_conduits = network_data['start_conduits_random_per_component']
        cfg['start_conduits'] = start_conduits

    cfg['conduit_diameters'] = 'inherit_from_net'

    bubble_expansion_probabilities = percolation.get_spontaneous_embolism_probability(vulnerability_pressure_range)
    cfg['bubble_expansion_probabilities'] = bubble_expansion_probabilities


    percolation.run_spreading_iteration(net, cfg, pressure, save_path, 
                                        spreading_probability_range=probability, 
                                        si_length=cfg['si_length'], include_orig_values=True)

