"""
A script for visualizing vulnerability curve calculated using the physiological spreading model and optimized SI spreading probabilities.
This script produces Fig. 6 of the manuscript.

Written by Onerva Korhonen
"""
import params
import visualization

import pickle
import numpy as np
import matplotlib.pylab as plt

if __name__=='__main__':
    data_save_folders = [base.rsplit('/', 1)[0] for base in params.optimized_vc_plot_data_save_path_bases]
    visualization.plot_optimized_vulnerability_curve(data_save_folders, params.physiological_vc_color, params.stochastic_vc_color, 
                                                     params.physiological_vc_alpha, params.stochastic_vc_alpha, params.optimized_vc_linestyles, params.optimized_vc_labels,
                                                     params.optimized_vc_plot_vs_empirical_save_path, pooled_data=True, pooled_data_save_name=params.pooled_optimized_spreading_probability_vs_empirical_save_name,
                                                     std_alpha=params.std_alpha, prevalence_linestyles=params.prevalence_linestyles,
                                                     prevalence_plot_save_path_base=params.optimized_prevalence_plot_vs_empirical_save_path_base)
