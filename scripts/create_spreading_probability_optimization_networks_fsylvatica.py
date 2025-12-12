"""
A script for creating a set of networks for studying embolism spreading at different pressure differences in F. sylvatica.

After this, run create_spreading_probability_optimization_data.py for creating the data needed for figures corresponding to those of 3-5 of the SI spreading manuscript.

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
cfg['spontaneous_embolism'] = params.spontaneous_embolism
cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path

cfg['NPc'] = fsylvatica_params.NPc
cfg['Pc'] = fsylvatica_params.Pc
cfg['Pe_rad'] = fsylvatica_params.Pe_rad
cfg['Pe_tan'] = fsylvatica_params.Pe_tan

# parameters specific to this run
cfg['net_size'] = [100, 10, 100]
cfg['bpp_type'] = 'young-laplace_with_constrictions'
cfg['spontaneous_embolism'] = False

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

percolation_type = 'si'
removal_order = 'random'
break_nonfunctional_components = False
cec_indicator = mrad_params.cec_indicator

# NOTE: do not modify any parameters below this point

if __name__=='__main__':

    index = int(sys.argv[1])
   
    n_segments = len(fsylvatica_params.Dc)
    
    iteration_index = int(np.floor(index / n_segments))
    segment_index = index - iteration_index * n_segments
    
    cfg['Dc'] = fsylvatica_params.Dc[segment_index]
    cfg['Dc_cv'] = fsylvatica_params.Dc_cv[segment_index]
    cfg['fc'] = fsylvatica_params.fc[segment_index]
    cfg['fpf'] = fsylvatica_params.fpf[segment_index]
    cfg['segment_name'] = fsylvatica_params.segment_names[segment_index]

    save_path = params.spreading_probability_optimization_network_save_path_base + '_' + fsylvatica_params.segment_names[segment_index] + '_' + str(iteration_index) + '.pkl'

    cfg['conduit_diameters'] = 'lognormal'

    # creating the xylem network following Mrad and preparing it for simulations with OpenPN
    conduit_elements, conns = mrad_model.create_mrad_network(cfg)
    net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
    net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
    mrad_model.prepare_simulation_network(net_cleaned, cfg, update_coords=True)

    conns = net_cleaned['throat.conns']
    conn_types = net_cleaned['throat.type']
    cec_mask = conn_types == cec_indicator
    cec = conns[cec_mask]
    conduits = mrad_model.get_conduits(cec)

    _, component_indices, _ = mrad_model.get_components(net_cleaned)
    start_conduits = np.zeros(len(component_indices), dtype=int)
    for i, component_elements in enumerate(component_indices):
        start_element = np.random.choice(component_elements)
        start_conduits[i] = np.where((conduits[:, 0] <= start_element) & (start_element <= conduits[:, 1]))[0][0]

    network_data = {'network':net_cleaned, 'start_conduits_random_per_component':start_conduits}

    f = open(save_path, 'wb')
    pickle.dump(network_data, f)
    f.close()
