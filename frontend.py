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

import openpnm as op

cfg = {}
cfg['net_size'] = mrad_params.net_size

conduit_elements, conns = mrad_model.create_mrad_network(cfg) # if no params are given, the function uses the default params of the Mrad model
net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
mrad_model.visualize_network_with_openpnm(net)
net_cleaned = mrad_model.clean_network(net, conduit_elements, cfg['net_size'][0] - 1)
mrad_model.visualize_network_with_openpnm(net_cleaned)

mrad_model.save_network(net_cleaned, params.network_save_file)

