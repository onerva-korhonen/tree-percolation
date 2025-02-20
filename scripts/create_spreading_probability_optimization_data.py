"""
A script for creating data for optimizing SI spreading probability (this means simulating embolism spreading for a set of pressure differences and spreading probabilities;
the evolution of effective conductance and network properties are saved for each pressure difference / spreading probability.

This script creates the data needed for Figs. 3-5 of the manuscript.

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
cfg['Dp'] = params.Dp
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
cfg['net_size'] = [100, 100, 100]
cfg['bpp_type'] = 'young-laplace_with_constrictions'
cfg['spontaneous_embolism'] = False

vulnerability_pressure_range = np.arange(0, 8, step=0.25)*1E6
spreading_probability_range = np.logspace(np.log10(0.0001), np.log10(0.02), 15)

include_orig_values = True

cfg['use_cylindrical_coords'] = False
cfg['fixed_random'] = False

visualize_simulations = False

percolation_type = 'si'
removal_order = 'random'
break_nonfunctional_components = False

#n_iterations = 100
pressures = [[pressure] for pressure in vulnerability_pressure_range] + [[] for i in range(len(spreading_probability_range))]
probabilities = [[] for i in range(len(vulnerability_pressure_range))] + [[probability] for probability in spreading_probability_range]
n_pressures = len(pressures)

# NOTE: do not modify any parameters below this point

if __name__=='__main__':

    index = int(sys.argv[1])
    iteration_index = int(np.floor(index / n_pressures))

    pressure = pressures[index - n_pressures * iteration_index]
    probability = probabilities[index - n_presssures * iteration_index]
    save_path = params.optimized_spreading_probability_save_path_base + '_' + str(index) + '.pkl'

    cfg['conduit_diameters'] = 'lognormal'

    # creating the xylem network following Mrad and preparing it for simulations with OpenPN
    conduit_elements, conns = mrad_model.create_mrad_network(cfg)
    net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
    net_cleaned, _ = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
    mrad_model.prepare_simulation_network(net_cleaned, cfg, update_coords=True)

    cfg['conduit_diameters'] = 'inherit_from_net'

    percolation.run_spreading_iteration(net, cfg, pressure, save_path, 
                                        spreading_probability_range=probability, 
                                        si_length=cfg['si_length'], include_orig_values=True)

