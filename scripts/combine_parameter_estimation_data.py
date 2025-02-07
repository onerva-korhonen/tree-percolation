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

average_diameter = params.Dc
average_area = np.pi * (average_diameter/2)**2
target_density = params.target_conduit_density * average_area # transferring the 1/m^2 conduit density from Lintunen & Kalliokoski 2010 to a fraction of occupied cells
target_length = 0 # TODO: find average conduit length for Betula pendula
target_grouping_index = params.target_grouping_index
                                        
optimization_data_save_folder = params.parameter_optimization_save_path_base.rsplit('/', 1)[0]
optimization_data_save_name_base = params.parameter_optimization_save_path_base.rsplit('/', 1)[1]

if __name__=='__main__':
    NPc, Pc, Pe_rad, Pe_tan, achived_density, achieved_length, achieved_grouping_index = parameter_estimation.optimize_parameters_from_data(target_density, target_length, 
                                                                                                                                            target_grouping_index, 
                                                                                                                                            optimization_data_save_folder, 
                                                                                                                                            optimization_data_save_name_base,
                                                                                                                                            params.param_optimization_fig_save_path_base)
    print('Optimal NPc: ' + str(NPc))
    print('Optimal Pc:' + str(Pc))
    print('Optimal Pe_rad:' + str(Pe_rad))
    print('Optimal Pe_tan:' + str(Pe_tan))
    print('Target conduit density ' + str(target_density) + ', achieved conduit density ' + str(achived_density))
    print('Target conduit length ' + str(target_length) + ', achieved conduit length ' + str(achieved_length))
    print('Target grouping index: ' + str(target_grouping_index) + ', achieved grouping index: ' + str(achieved_grouping_index))
