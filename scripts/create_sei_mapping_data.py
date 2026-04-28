#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:01:47 2026

@author: Onerva Korhonen

A script for creating data for mapping the SEI spreading outcome space spanned by
the two transition probabilities (S-E probability and E-I probability).
"""
import percolation
import params

import numpy as np
import sys

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

include_orig_values = False

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

percolation_type = 'si'
removal_order = 'random'
break_nonfunctional_components = False

create_networks = False
project_specific_networks = False # set to True for using networks created for the current project, False to re-used networks created for an earlier project
create_expansion_probability_data = False

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
    
    se_probability = se_probabilities[se_index]
    ei_probability_range = ei_probabilities[slice_index * int(np.floor(n_ei_probabilities / n_slices)) : (slice_index + 1) *  int(np.floor(n_ei_probabilities / n_slices))]
    if slice_index == n_slices - 1:
        np.concat((ei_probability_range, ei_probabilities[(slice_index + 1) *  int(np.floor(n_ei_probabilities / n_slices))::]))
        
    # pseudocode:
        # for each ei_probability in ei_probability_range:
        #    run_conduit_si with relevant parameters
        #    save outputs (final effective conductance and simulation length); for this, need to define save path in params

    # requires a specific script for reading and visualizing the data
    # is it better to keep this as a script or write these as functions?