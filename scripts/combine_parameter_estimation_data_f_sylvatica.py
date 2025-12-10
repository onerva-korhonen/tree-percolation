"""
A script for combining parameter estimation data and estimating optimal probability parameters for Fagus sylvatica.
"""

import numpy as np
import sys
import os
import pickle
import matplotlib.pylab as plt
from scipy.ndimage import label, generate_binary_structure

import mrad_model
import mrad_params
import params
import parameter_estimation

average_diameter = params.Dc_f_sylvatica
average_area = np.pi * (average_diameter/2)**2
target_density = params.target_conduit_density_f_sylvatica * average_area # transferring the 1/m^2 conduit density to a fraction of occupied cells
target_length = 0 # TODO: find average conduit length for F. sylvatica
target_grouping_index = params.target_grouping_index_f_sylvatica

exclude_nonconsistents = True # To exclude from optimizations parameter combinations that produced at least one empty network
                                        
optimization_data_save_folder = params.parameter_optimization_save_path_base.rsplit('/', 1)[0]
optimization_data_save_name_base = params.parameter_optimization_save_path_base.rsplit('/', 1)[1]

if __name__=='__main__':
    NPc, Pc, Pe_rad, Pe_tan, achived_density, achieved_length, achieved_grouping_index = parameter_estimation.optimize_parameters_from_data(target_density, target_length, 
                                                                                                                                            target_grouping_index, 
                                                                                                                                            optimization_data_save_folder, 
                                                                                                                                            optimization_data_save_name_base,
                                                                                                                                            params.param_optimization_fig_save_path_base,
                                                                                                                                            exclude_nonconsistents=exclude_nonconsistents)
    print('Optimal NPc: ' + str(NPc))
    print('Optimal Pc:' + str(Pc))
    print('Optimal Pe_rad:' + str(Pe_rad))
    print('Optimal Pe_tan:' + str(Pe_tan))
    print('Target conduit density ' + str(target_density) + ', achieved conduit density ' + str(achived_density))
    print('Target conduit length ' + str(target_length) + ', achieved conduit length ' + str(achieved_length))
    print('Target grouping index: ' + str(target_grouping_index) + ', achieved grouping index: ' + str(achieved_grouping_index))
