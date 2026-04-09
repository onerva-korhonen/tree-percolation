#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 8 14:08 2026

@author: Onerva Korhonen

This is script for pooling physiological spreading simulation data (with no stochastic spreading probability optimization, at least so far). A test script for the SEI spreading.

After running this, run visualize_optimized_vc.py and visualize_single_param_spreading.py
"""
import params
import percolation

import sys

max_n_iterations = 100
pool_physiological_only = True

if __name__=='__main__':
   
    simulation_data_save_folder = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[0]
    simulation_data_save_name_base = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[1]
    pooled_data_save_path = simulation_data_save_folder + '/' + params.pooled_optimized_spreading_probability_save_name
    
    percolation.optimize_spreading_probability_from_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, max_n_iterations=max_n_iterations,
                                                         pool_physiological_only=pool_physiological_only)
    


