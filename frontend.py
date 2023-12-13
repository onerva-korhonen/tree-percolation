#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:43:35 2023

@author: onerva

A frontend script for investigating percolation using the Mrad et al. xylem network model
"""
import mrad_model
import mrad_params
import params
import visualization
import percolation

import openpnm as op
import numpy as np

cfg = {}
cfg['net_size'] = mrad_params.net_size
cfg['conduit_diameters'] = 'lognormal'#mrad_params.conduit_diameters
cfg['Dc'] = mrad_params.Dc
cfg['Dc_cv'] = mrad_params.Dc_cv
cfg['seeds_NPc'] = mrad_params.seeds_NPc
cfg['seeds_Pc'] = mrad_params.seeds_Pc
cfg['seed_ICC_rad'] = mrad_params.seed_ICC_rad
cfg['seed_ICC_tan'] = mrad_params.seed_ICC_tan
cfg['si_type'] = params.si_type
cfg['si_length'] = params.si_length
cfg['start_conduit'] = params.start_conduit
cfg['spreading_probability'] = params.spreading_probability
cfg['spreading_threshold'] = params.spreading_threshold

cfg['fixed_random'] = True

conduit_elements, conns = mrad_model.create_mrad_network(cfg) # if no params are given, the function uses the default params of the Mrad model
net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
visualization.visualize_network_with_openpnm(net)
net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
visualization.visualize_network_with_openpnm(net_cleaned)

#mrad_model.save_network(net_cleaned, params.network_save_file)

sim_net = mrad_model.prepare_simulation_network(net_cleaned, cfg)
visualization.visualize_pores(sim_net)
visualization.visualize_network_with_openpnm(sim_net, params.use_cylindrical_coordinates, mrad_params.Lce, 'pore.coords')
effective_conductance = mrad_model.simulate_water_flow(sim_net, cfg, visualize=params.visualize_simulations)
if params.percolation_type in ['conduit', 'si']:
    lcc_size, susceptibility, _ = percolation.get_conduit_lcc_size(sim_net)
else:
    lcc_size, susceptibility = percolation.get_lcc_size(sim_net)
n_inlet, n_outlet = percolation.get_n_inlets(sim_net, cfg['net_size'][0] - 1, use_cylindrical_coords=True)

cfg['use_cylindrical_coords'] = False
net_cleaned['pore.diameter'] = sim_net['pore.diameter']
effective_conductances, lcc_sizes, functional_lcc_sizes, nonfunctional_component_size, susceptibilities, functional_susceptibilities, n_inlets, n_outlets, nonfunctional_component_volume, prevalence = percolation.run_percolation(net_cleaned, cfg, percolation_type=params.percolation_type, removal_order='random', break_nonfunctional_components=params.break_nonfunctional_components)
effective_conductances = np.append(np.array([effective_conductance]), effective_conductances)
lcc_sizes = np.append(np.array([lcc_size]), lcc_sizes)
functional_lcc_sizes = np.append(np.array([lcc_size]), functional_lcc_sizes)
nonfunctional_component_size = np.append(np.array([0]), nonfunctional_component_size)
susceptibilities = np.append(np.array([susceptibility]), susceptibilities)
functional_susceptibilities = np.append(np.array([susceptibility]), functional_susceptibilities)
n_inlets = np.append(np.array([n_inlet]), n_inlets)
n_outlets = np.append(np.array([n_outlet]), n_outlets)
nonfunctional_component_volume = np.append(np.array([0]), nonfunctional_component_volume)
percolation_outcome_values = np.concatenate((np.expand_dims(effective_conductances, axis=0), 
                                             np.expand_dims(lcc_sizes, axis=0), np.expand_dims(functional_lcc_sizes, axis=0)),
                                             axis=0)
if params.percolation_type in ['conduit', 'si']:
    total_n_nodes = len(effective_conductances)
elif params.percolation_type == 'bond':
    total_n_nodes = net_cleaned['throat.conns'].shape[0] + 1
elif params.percolation_type == 'site':
    total_n_nodes = net_cleaned['pore.coords'].shape[0] + 1
else:
    raise Exception('Unknown percolation type; percolation type must be bond, site, or conduit')
if params.percolation_type == 'si':
    x = np.append(np.array([0]), prevalence)
else:
    x = []
visualization.plot_percolation_curve(total_n_nodes, percolation_outcome_values,
                                     colors=params.percolation_outcome_colors, labels=params.percolation_outcome_labels, 
                                     alphas=params.percolation_outcome_alphas, y_labels=params.percolation_outcome_ylabels,
                                     axindex=params.percolation_outcome_axindex, save_path=params.percolation_plot_save_path, x=x)
visualization.plot_percolation_curve(total_n_nodes, np.expand_dims(nonfunctional_component_volume, axis=0),
                                     colors=[params.percolation_nonfunctional_component_size_color], labels=[params.percolation_nonfunctional_component_size_label], 
                                     alphas=[params.percolation_nonfunctional_component_size_alpha], save_path=params.nonfunctional_componen_size_save_path, x=x)
visualization.plot_percolation_curve(total_n_nodes, 
                                     np.concatenate((np.expand_dims(n_inlets, axis=0), np.expand_dims(n_outlets, axis=0)), axis=0),
                                     colors=[params.percolation_ninlet_color, params.percolation_noutlet_color],
                                     labels=[params.percolation_ninlet_label, params.percolation_noutlet_label],
                                     alphas=[params.percolation_ninlet_alpha, params.percolation_noutlet_alpha],
                                     save_path=params.ninlet_save_path, x=x)
