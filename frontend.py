#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:43:35 2023

@author: onerva

A frontend script for investigating percolation using the Mrad et al. xylem network model
"""
import mrad_model
import mrad_params

conduits, conns = mrad_model.create_mrad_network(cfg={}) # if no params are given, the function uses the default params of the Mrad model
mrad_model.visualize_network_with_openpnm(conduits, conns)


