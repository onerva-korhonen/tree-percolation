#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 8 17:45 2026

@author: Onerva Korhonen

A script for visualizing the physiological vulnerability curve simulated with the "exposed" group ("physiological SEI").
"""
import params
import visualization

if __name__=='__main__':

    data_save_folders = [base.rsplit('/', 1)[0] for base in params.optimized_vc_plot_data_save_path_bases]

    visualization.plot_optimized_vulnerability_curve(data_save_folders, params.physiological_vc_color, params.stochastic_vc_color, 
            params.physiological_vc_alpha, params.stochastic_vc_alpha, params.optimized_vc_linestyles, params.optimized_vc_labels,
                                                     params.p_50_color, params.p_50_alpha, params.p_50_line_style,
                                                     params.optimized_vc_plot_save_path, pooled_data=True, pooled_data_save_name=params.pooled_optimized_spreading_probability_save_name,
                                                     std_alpha=params.std_alpha, prevalence_linestyles=params.prevalence_linestyles,
                                                     prevalence_plot_save_path_base=params.optimized_prevalence_plot_save_path_base, 
                                                     upper_pressure_limit=params.optimized_vc_upper_pressure_limit,
                                                     physiological_only=True)

