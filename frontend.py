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
cfg['conduit_element_length'] = params.Lce
cfg['fc'] = params.fc
cfg['average_pit_area'] = params.average_pit_membrane_area
cfg['fpf'] = params.fpf
cfg['tf'] = params.tf
cfg['Dp'] = params.Dp
cfg['Tm'] = params.Tm
cfg['weibull_a'] = params.weibull_a
cfg['weibull_b'] = params.weibull_b
cfg['n_constrictions'] = params.n_constrictions
cfg['truncnorm_center'] = params.truncnorm_center
cfg['truncnorm_std'] = params.truncnorm_std
cfg['truncnorm_a'] = params.truncnorm_a
cfg['pore_shape_correction'] = params.pore_shape_correction
cfg['gas_contact_angle'] = params.gas_contact_angle
cfg['icc_length'] = params.icc_length
cfg['seeds_NPc'] = params.seeds_NPc
cfg['seeds_Pc'] = params.seeds_Pc
cfg['seed_ICC_rad'] = params.seed_ICC_rad
cfg['seed_ICC_tan'] = params.seed_ICC_tan
cfg['si_length'] = params.si_length
cfg['si_tolerance_length'] = params.si_tolerance_length
cfg['si_type'] = params.si_type
cfg['start_conduits'] = params.start_conduits
cfg['surface_tension'] = params.surface_tension
cfg['pressure'] = params.pressure
cfg['nCPUs'] = params.nCPUs
cfg['spontaneous_embolism'] = params.spontaneous_embolism
cfg['bpp_type'] = params.bpp_type
cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path
# TODO: check that the followig params match the physiology of betula pendula
#cfg['weibull_a'] = mrad_params.weibull_a
#cfg['weibull_b'] = mrad_params.weibull_b

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = True

simulate_single_param_spreading = False
construct_VC = False
optimize_spreading_probability = False
create_optimization_data = False
combine_optimization_data = True
time = False

#print(cfg['net_size'])
if __name__=='__main__':

    conduit_elements, conns = mrad_model.create_mrad_network(cfg) # if no params are given, the function uses the default params of the Mrad model
    net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
    #visualization.visualize_network_with_openpnm(net)
    net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
    #visualization.visualize_network_with_openpnm(net_cleaned)
    orig_n_throats = net_cleaned['throat.conns'].shape[0] + 1
    orig_n_pores = net_cleaned['pore.coords'].shape[0] + 1

    #mrad_model.save_network(net_cleaned, params.network_save_file)

    mrad_model.prepare_simulation_network(net_cleaned, cfg, update_coords=True)
    #visualization.visualize_pores(sim_net)
    #visualization.visualize_network_with_openpnm(sim_net, params.use_cylindrical_coordinates, mrad_params.Lce, 'pore.coords')

    if simulate_single_param_spreading:
        
        cfg['spreading_probability'] = params.spreading_probability
        cfg['pressure_diff'] = params.pressure_diff
        effective_conductance, _ = simulations.simulate_water_flow(net_cleaned, cfg, visualize=params.visualize_simulations)
    
        if params.percolation_type in ['conduit', 'si']:
            lcc_size, susceptibility, _ = percolation.get_conduit_lcc_size(net=net_cleaned)
        else:
            lcc_size, susceptibility = percolation.get_lcc_size(net_cleaned)
        n_inlet, n_outlet = percolation.get_n_inlets(net_cleaned, (cfg['net_size'][0] - 1)*cfg['conduit_element_length'], use_cylindrical_coords=False)
    
        cfg['use_cylindrical_coords'] = False
        cfg['conduit_diameters'] = 'inherit_from_net'
        
        effective_conductances, lcc_sizes, functional_lcc_sizes, nonfunctional_component_size, susceptibilities, functional_susceptibilities, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading = percolation.run_percolation(net_cleaned, cfg, percolation_type=params.percolation_type, removal_order='random', break_nonfunctional_components=params.break_nonfunctional_components)
        effective_conductances = np.append(np.array([effective_conductance]), effective_conductances)
        lcc_sizes = np.append(np.array([lcc_size]), lcc_sizes)
        functional_lcc_sizes = np.append(np.array([lcc_size]), functional_lcc_sizes)
        nonfunctional_component_size = np.append(np.array([0]), nonfunctional_component_size)
        susceptibilities = np.append(np.array([susceptibility]), susceptibilities)
        functional_susceptibilities = np.append(np.array([susceptibility]), functional_susceptibilities)
        n_inlets = np.append(np.array([n_inlet]), n_inlets)
        n_outlets = np.append(np.array([n_outlet]), n_outlets)
        nonfunctional_component_volume = np.append(np.array([0]), nonfunctional_component_volume)
        prevalence_due_to_spontaneous_embolism = np.append(np.array([0]), prevalence_due_to_spontaneous_embolism)
        prevalence_due_to_spreading = np.append(np.array([0]), prevalence_due_to_spreading)
        percolation_outcome_values = np.concatenate((np.expand_dims(effective_conductances, axis=0), 
                                                     np.expand_dims(lcc_sizes, axis=0), np.expand_dims(functional_lcc_sizes, axis=0)),
                                                     axis=0)
        if params.percolation_type in ['conduit', 'si', 'drainage']:
            total_n_nodes = len(effective_conductances)
        elif params.percolation_type == 'bond':
            total_n_nodes = orig_n_pores
        elif params.percolation_type == 'site':
            total_n_nodes = orig_n_throats
        else:
            raise Exception('Unknown percolation type; percolation type must be bond, site, conduit, si or drainage')
        if params.percolation_type in ['si', 'drainage']:
            x = np.append(np.array([0]), prevalence)
        else:
            x = []
    
        data = {'effective_conductances':effective_conductances, 'lcc_sizes':lcc_sizes, 'functional_lcc_sizes':functional_lcc_sizes, 
                'nonfunctional_component_size':nonfunctional_component_size, 'susceptibilities':susceptibilities, 
                'functional_susceptibilities':functional_susceptibilities, 'n_inlets':n_inlets, 'n_outlets': n_outlets, 
                'nonfunctional_component_volume':nonfunctional_component_volume, 'total_n_nodes':total_n_nodes, 'x':x,
                'prevalence_due_to_spontaneous_embolism':prevalence_due_to_spontaneous_embolism, 'prevalence_due_to_spreading':prevalence_due_to_spreading}
        f = open(params.percolation_data_save_path, 'wb')
        pickle.dump(data, f)
        f.close()
    
    if construct_VC:
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
        if params.spontaneous_embolism:
            spontaneous_embolism_probabilities = percolation.get_spontaneous_embolism_probability(params.vulnerability_pressure_range)
            cfg['spontaneous_embolism_probabilities'] = spontaneous_embolism_probabilities
            
        index = int(sys.argv[1])
        pressure_diff = params.vulnerability_pressure_range[index]

        cfg['conduit_diameters'] = 'inherit_from_net'

        percolation.optimize_spreading_probability(net_cleaned, cfg, pressure_diff, params.optimization_probability_range, si_length=cfg['si_length'], n_iterations=params.n_iterations, save_path_base=params.optimized_spreading_probability_save_path_base)
    
    if create_optimization_data:
        if params.spontaneous_embolism:
            spontaneous_embolism_probabilities = percolation.get_spontaneous_embolism_probability(params.vulnerability_pressure_range)
            cfg['spontaneous_embolism_probabilities'] = spontaneous_embolism_probabilities
            
        cfg['conduit_diameters'] = 'inherit_from_net'
        
        index = int(sys.argv[1])
        save_path = params.optimized_spreading_probability_save_path_base + '_' + str(index) + '.pkl'
        percolation.run_spreading_iteration(net, cfg, params.vulnerability_pressure_range, save_path, 
                                            spreading_probability_range=params.optimization_probability_range, si_length=cfg['si_length'])
    if combine_optimization_data:
        simulation_data_save_folder = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[0]
        simulation_data_save_name_base = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[1]
        pooled_data_save_path = simulation_data_save_folder + '/' + params.pooled_optimized_spreading_probability_save_name
        percolation.optimize_spreadig_probability_from_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path)
    
    if time:
        import timeit
        
        setup = """
        
import mrad_model
import mrad_params
import params
from percolation import calculate_bpp, get_conduit_neighbors

import openpnm as op
import numpy as np

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

heartwood_d = cfg.get('heartwood_d', mrad_params.heartwood_d)

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = True

conduit_elements, conns = mrad_model.create_mrad_network(cfg) # if no params are given, the function uses the default params of the Mrad model
net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
mrad_model.prepare_simulation_network(net_cleaned, cfg, update_coords=True)
        
cec_indicator = cfg.get('cec_indicator', mrad_params.cec_indicator)
throats = net_cleaned['throat.conns']
throat_types = net_cleaned['throat.type']
cec_mask = throat_types == cec_indicator
cec = throats[cec_mask]
conduits = mrad_model.get_conduits(cec)
conduit_neighbours = get_conduit_neighbors(net_cleaned, cfg['use_cylindrical_coords'], cfg['conduit_element_length'], heartwood_d, cec_indicator)
"""

        code_to_time = """
bpp = calculate_bpp(net_cleaned, conduits, 1 - cec_mask, cfg)
conduit_neighbour_bpp = {}
for i, conduit in enumerate(conduits):
    iccs = throats[np.where(1 - cec_mask)]
    conduit_iccs = np.where(((conduit[0] <= iccs[:, 0]) & (iccs[:, 0] <= conduit[1])) | ((conduit[0] <= iccs[:, 1]) & (iccs[:, 1] <= conduit[1])))[0]
    neighbours = conduit_neighbours[i]
    neighbour_bpps = {}
    for neighbour in neighbours:
        neighbour_iccs = np.where(((conduits[neighbour, 0] <= iccs[:, 0]) & (iccs[:, 0] <= conduits[neighbour, 1])) | ((conduits[neighbour, 0] <= iccs[:, 1]) & (iccs[:, 1] <= conduits[neighbour, 1])))[0]
        shared_iccs = np.intersect1d(conduit_iccs, neighbour_iccs)
        shared_bpp = bpp[shared_iccs]
        neighbour_bpps[neighbour] = np.amin(shared_bpp)
    conduit_neighbour_bpp[i] = neighbour_bpps
"""
        
        exec_time = timeit.timeit(setup=setup, stmt=code_to_time,number=1000000) * 10**3
        print(f"The time of execution of extracting bpp's between conduit pairs' : {exec_time:.03f}ms")
        

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
