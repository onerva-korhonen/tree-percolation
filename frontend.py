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

import openpnm as op

cfg = {}
cfg['net_size'] = mrad_params.net_size

conduit_elements, conns = mrad_model.create_mrad_network(cfg) # if no params are given, the function uses the default params of the Mrad model
net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
mrad_model.visualize_network_with_openpnm(net)
net_cleaned = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
visualization.visualize_network_with_openpnm(net_cleaned)

mrad_model.save_network(net_cleaned, params.network_save_file)

sim_net = mrad_model.prepare_simulation_network(net_cleaned, cfg)
visualization.visualize_pores(sim_net)
visualization.visualize_network_with_openpnm(sim_net, params.use_cylindrical_coordinates, mrad_params.Lce, 'pore.coords')
effective_conductance = mrad_model.simulate_water_flow(sim_net, cfg, visualize=params.visualize_simulations)
print(effective_conductance)
