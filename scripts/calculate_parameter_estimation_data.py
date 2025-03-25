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

in_iterations = 10

start_divider = 1210 # len(end_range) * len(Pe_rad_range) * n_iterations
end_divider = 110 # len(Pe_rad_range) * n_iterations
rad_divider = 10 # n_iterations

start_range = np.arange(0, 1.1, 0.1) 
end_range = np.arange(0, 1.1, 0.1)

Pe_rad_range = np.arange(0, 1.1, 0.1)
Pe_tan_range = np.arange(0, 1.1, 0.1)

Pes_rad = [[Pe_rad, Pe_rad] for Pe_rad in Pe_rad_range]
Pes_tan = [[Pe_tan, Pe_tan] for Pe_tan in Pe_tan_range]

optimization_net_size = [100, 10, 100]

if __name__=='__main__':
    index = int(sys.argv[1])

    start_index = int(np.floor(index / start_divider))
    end_index = int(np.floor((index - start_index * start_divider) / end_divider))
    rad_index = int(np.floor((index - start_index * start_divider - end_index * end_divider) / rad_divider))

    start_range = np.array([start_range[start_index]])
    end_range = np.array([end_range[end_index]])
    Pes_rad = [Pes_rad[rad_index]]

    save_path = params.parameter_optimization_save_path_base + '_' + str(index) + '.pkl'

    parameter_estimation.run_parameter_optimization_iteration(save_path, conduit_element_length=mrad_params.Lce, optimization_net_size=optimization_net_size, 
                                         start_range=start_range, end_range=end_range, Pes_rad=Pes_rad, Pes_tan=Pes_tan)
