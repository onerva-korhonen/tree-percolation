#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:01:47 2026

@author: Onerva Korhonen

A script for creating data for mapping the SEI spreading outcome space spanned by
the two transition probabilities (S-E probability and E-I probability).
"""
import numpy as np
import sys
import pickle
import os

import percolation
import params
import mrad_model

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
cfg['delayed_embolism'] = True
cfg['si_type'] = 'stochastic_sei'

include_orig_values = True

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

create_networks = False
project_specific_networks = False # set to True for using networks created for the current project, False to re-used networks created for an earlier project

project_specific_bpp = False
if not project_specific_bpp:
    cfg['bpp_data_path'] = params.alternative_bubble_propagation_pressure_data_path # set to True for using BPP data created for the current project, False to re-used BPP data created for an earlier project

se_step = 0.01
ei_step = 0.01
se_probabilities = np.arange(0, 1 + se_step, se_step)
ei_probabilities = np.arange(0, 1 + ei_step, ei_step)

n_iterations = 100
n_se_probabilities = len(se_probabilities)
n_ei_probabilities = len(ei_probabilities)
n_slices = 4

# NOTE: do not modify any parameters below this point
# Note on the indexing order: calculations are performed in the iteration -> S-E probability -> E-I probability slice order (first one iteration for all slices of first S-E probability)

if __name__=='__main__':

    index = int(sys.argv[1])
    
    iteration_index = int(np.floor(index / (n_se_probabilities * n_slices)))
    se_index = int(np.floor((index - iteration_index * n_se_probabilities * n_slices) / n_slices))
    slice_index = index - iteration_index * n_se_probabilities * n_slices - se_index * n_slices
    
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
    
    se_probability = se_probabilities[se_index]
    ei_probability_range = ei_probabilities[slice_index * int(np.floor(n_ei_probabilities / n_slices)) : (slice_index + 1) *  int(np.floor(n_ei_probabilities / n_slices))]
    if slice_index == n_slices - 1:
        ei_probability_range = np.concat((ei_probability_range, ei_probabilities[(slice_index + 1) *  int(np.floor(n_ei_probabilities / n_slices))::]))
        
    plcs = np.zeros(len(ei_probability_range))
    simulation_lengths = np.zeros(len(ei_probability_range))

    for i, ei_probability in enumerate(ei_probability_range):
        cfg['bubble_expansion_probability'] = ei_probability
        effective_conductances, _, _, _, _, _, _, _, _, _, _, _, _, _ = percolation.run_conduit_si(net, cfg, se_probability, include_orig_values)
        plcs[i] = 100 * (1 - effective_conductances[-1] / effective_conductances[0]) # final effective conductance divided by the effective conductance of the intact network
        simulation_lengths[i] = len(effective_conductances) - 1 # - 1 compensates for including the effective conductance of the intact network as the first value
        
    data = {'se_probability':se_probability, 'ei_probabilities':ei_probability_range, 'plcs':plcs, 'simulation_lengths':simulation_lengths}
    
    save_path = params.sei_mapping_data_save_path_base + '_' + str(index) + '.pkl'
    
    save_folder = save_path.rsplit('/', 1)[0]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()

