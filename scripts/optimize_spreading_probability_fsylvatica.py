#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:34:33 2025

@author: Onerva Korhonen

This is script for reading spreading simulation data for F. sylvatica (or other species with multiple segments measured).

After running this, run visualize_optimized_vc.py and visualize_single_param_spreading.py
"""
import params
import fsylvatica_params
import percolation

import sys

max_n_iterations = 1

segment_names = fsylvatica_params.segment_names

if __name__=='__main__':
    
    index = int(sys.argv[1])
    
    segment_name = segment_names[index]
    
    simulation_data_save_folder = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[0]
    simulation_data_save_name_base = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[1] + '_' + segment_name
    pooled_data_save_path = simulation_data_save_folder + '/' + params.pooled_optimized_spreading_probability_save_name.split('.')[0] + '_' + segment_name + 'pkl'
    
    percolation.optimize_spreading_probability_from_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, max_n_iterations=max_n_iterations)
    


