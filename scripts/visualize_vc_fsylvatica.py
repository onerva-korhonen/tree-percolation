#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 09:53:20 2025

@author: Onerva Korhonen

A script for visualizing the physiological vulnerability curve for different stem segments for F. sylvatica (and other species, for which segment-specific simulations
have been performed.)
"""
import fsylvatica_params
import params
import visualization

if __name__=='__main__':

    data_save_folders = fsylvatica_params.spreading_simulation_data_folders
    optimized_vc_labels = fsylvatica_params.segment_names
    vc_linestyles = fsylvatica_params.vc_linestyles
    pooled_data_save_name = [params.pooled_optimized_spreading_probability_save_name.split('.')[0] + '_' + segment_name + '.pkl' for segment_name in fsylvatica_params.segment_names]
    
    visualization.plot_optimized_vulnerability_curve(data_save_folders, params.physiological_vc_color, params.stochastic_vc_color, 
                                                     params.physiological_vc_alpha, params.stochastic_vc_alpha, vc_linestyles, optimized_vc_labels,
                                                     params.p_50_color, params.p_50_alpha, params.p_50_line_style,
                                                     params.optimized_vc_plot_save_path, pooled_data=True, pooled_data_save_name=pooled_data_save_name,
                                                     std_alpha=params.std_alpha, prevalence_linestyles=params.prevalence_linestyles,
                                                     prevalence_plot_save_path_base=params.optimized_prevalence_plot_save_path_base, 
                                                     upper_pressure_limit=params.optimized_vc_upper_pressure_limit,
                                                     pressures_to_be_visualized=params.optimized_vc_pressures_to_be_visualized)

