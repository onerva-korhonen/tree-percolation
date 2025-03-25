"""
A script for creating a set of networks for studying embolism spreading at different pressure differences / SI spreading probabilities.

After this, run create_spreading_probability_optimization_data.py for creating the data needed for figs. 3-5 of the manuscript.

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
cfg['spontaneous_embolism'] = params.spontaneous_embolism
cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path

cfg['NPc'] = params.NPc
cfg['Pc'] = params.Pc
cfg['Pe_rad'] = params.Pe_rad
cfg['Pe_tan'] = params.Pe_tan

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

n_networks = 100

# NOTE: do not modify any parameters below this point

if __name__=='__main__':

    index = int(sys.argv[1])

    save_path = params.spreading_probability_optimization_network_save_path_base + '_' + str(index) + '.pkl'

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
