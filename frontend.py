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
#import visualization
import percolation
import simulations

import openpnm as op
import numpy as np
import pickle
import sys

cfg = {}
cfg['net_size'] = params.net_size
cfg['conduit_diameters'] = 'lognormal'#mrad_params.conduit_diameters
cfg['Dc'] = params.Dc
cfg['Dc_cv'] = params.Dc_cv
cfg['seeds_NPc'] = params.seeds_NPc
cfg['seeds_Pc'] = params.seeds_Pc
cfg['seed_ICC_rad'] = params.seed_ICC_rad
cfg['seed_ICC_tan'] = params.seed_ICC_tan
cfg['si_length'] = params.si_length
cfg['si_tolerance_length'] = params.si_tolerance_length
cfg['si_type'] = params.si_type
cfg['start_conduits'] = params.start_conduits
cfg['air_contact_angle'] = params.air_contact_angle
cfg['surface_tension'] = params.surface_tension
cfg['pressure'] = params.pressure
cfg['nCPUs'] = params.nCPUs
# TODO: check that the followig params match the physiology of betula pendula
cfg['weibull_a'] = mrad_params.weibull_a
cfg['weibull_b'] = mrad_params.weibull_b
cfg['average_pit_area'] = mrad_params.Dm**2
cfg['conduit_element_length'] = mrad_params.Lce
cfg['fc'] = mrad_params.fc
cfg['fpf'] = mrad_params.fpf

cfg['fixed_random'] = True

simulate_single_param_spreading = False
construct_VC = False
optimize_spreading_probability = True

#print(cfg['net_size'])
if __name__=='__main__':

    conduit_elements, conns = mrad_model.create_mrad_network(cfg) # if no params are given, the function uses the default params of the Mrad model
    net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
    #visualization.visualize_network_with_openpnm(net)
    net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
    #visualization.visualize_network_with_openpnm(net_cleaned)

    #mrad_model.save_network(net_cleaned, params.network_save_file)

    sim_net = mrad_model.prepare_simulation_network(net_cleaned, cfg)
    #visualization.visualize_pores(sim_net)
    #visualization.visualize_network_with_openpnm(sim_net, params.use_cylindrical_coordinates, mrad_params.Lce, 'pore.coords')

    if simulate_single_param_spreading:
        cfg['spreading_probability'] = params.spreading_probability
        cfg['pressure_diff'] = params.pressure_diff
        effective_conductance, _ = simulations.simulate_water_flow(sim_net, cfg, visualize=params.visualize_simulations)
        if params.percolation_type in ['conduit', 'si']:
            lcc_size, susceptibility, _ = percolation.get_conduit_lcc_size(sim_net)
        else:
            lcc_size, susceptibility = percolation.get_lcc_size(sim_net)
        n_inlet, n_outlet = percolation.get_n_inlets(sim_net, cfg['net_size'][0] - 1, use_cylindrical_coords=True)
    
        cfg['use_cylindrical_coords'] = False
        net_cleaned['pore.diameter'] = sim_net['pore.diameter']
        cfg['conduit_diameters'] = 'inherit_from_net'
    
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
        if params.percolation_type in ['conduit', 'si', 'drainage']:
            total_n_nodes = len(effective_conductances)
        elif params.percolation_type == 'bond':
            total_n_nodes = net_cleaned['throat.conns'].shape[0] + 1
        elif params.percolation_type == 'site':
            total_n_nodes = net_cleaned['pore.coords'].shape[0] + 1
        else:
            raise Exception('Unknown percolation type; percolation type must be bond, site, conduit, si or drainage')
        if params.percolation_type in ['si', 'drainage']:
            x = np.append(np.array([0]), prevalence)
        else:
            x = []
    
        data = {'effective_conductances':effective_conductances, 'lcc_sizes':lcc_sizes, 'functional_lcc_sizes':functional_lcc_sizes, 'nonfunctional_component_size':nonfunctional_component_size, 'susceptibilities':susceptibilities, 'functional_susceptibilities':functional_susceptibilities, 'n_inlets':n_inlets, 'n_outlets': n_outlets, 'nonfunctional_component_volume':nonfunctional_component_volume, 'total_n_nodes':total_n_nodes, 'x':x}
        f = open(params.percolation_data_save_path, 'wb')
        pickle.dump(data, f)
        f.close()
    
    if construct_VC:
        cfg['use_cylindrical_coords'] = False
        net_cleaned['pore.diameter'] = sim_net['pore.diameter']
        cfg['conduit_diameters'] = 'inherit_from_net'
    
        if cfg['si_type'] == 'physiological':
            x_range = params.vulnerability_pressure_range
        elif cfg['si_type'] == 'stochastic':
            x_range = params.vulnerability_probability_range
        vc = percolation.construct_vulnerability_curve(net_cleaned, cfg, x_range, cfg['start_conduits'], cfg['si_length'])
    
        with open(params.vc_data_save_path, 'wb') as f:
            pickle.dump(vc, f)
        f.close()
    
    if optimize_spreading_probability:
        index = int(sys.argv[1])
        pressure_diff = params.vulnerability_pressure_range[index]

        cfg['use_cylindrical_coords'] = False
        net_cleaned['pore.diameter'] = sim_net['pore.diameter']
        cfg['conduit_diameters'] = 'inherit_from_net'
    
        optimized_spreading_probabilities = np.zeros(len(params.vulnerability_pressure_range))
        physiological_effective_conductances = np.zeros(len(params.vulnerability_pressure_range))
        stochastic_effective_conductances = np.zeros(len(params.vulnerability_pressure_range))
    
        percolation.optimize_spreading_probability(net_cleaned, cfg, pressure_diff, cfg['start_conduits'], params.optimization_probability_range, si_length=cfg['si_length'], n_iterations=params.n_iterations, save_path_base=params.optimized_spreading_probability_save_path_base)
        

#visualization.plot_percolation_curve(total_n_nodes, percolation_outcome_values,
#                                     colors=params.percolation_outcome_colors, labels=params.percolation_outcome_labels, 
#                                     alphas=params.percolation_outcome_alphas, y_labels=params.percolation_outcome_ylabels,
#                                     axindex=params.percolation_outcome_axindex, save_path=params.percolation_plot_save_path, x=x)
#visualization.plot_percolation_curve(total_n_nodes, np.expand_dims(nonfunctional_component_volume, axis=0),
#                                     colors=[params.percolation_nonfunctional_component_size_color], labels=[params.percolation_nonfunctional_component_size_label], 
#                                     alphas=[params.percolation_nonfunctional_component_size_alpha], save_path=params.nonfunctional_componen_size_save_path, x=x)
#visualization.plot_percolation_curve(total_n_nodes, 
#                                     np.concatenate((np.expand_dims(n_inlets, axis=0), np.expand_dims(n_outlets, axis=0)), axis=0),
#                                     colors=[params.percolation_ninlet_color, params.percolation_noutlet_color],
#                                     labels=[params.percolation_ninlet_label, params.percolation_noutlet_label],
#                                     alphas=[params.percolation_ninlet_alpha, params.percolation_noutlet_alpha],
#                                     save_path=params.ninlet_save_path, x=x)
