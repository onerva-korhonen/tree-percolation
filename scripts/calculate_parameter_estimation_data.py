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

start_divider = 1000 # len(end_range) * len(Pe_rad_range) * n_iterations
end_divider = 100 # len(Pe_rad_range) * n_iterations
rad_divider = 10 # n_iterations

#start_range = np.arange(0, 1.1, 0.1) 
#end_range = np.arange(0, 1.1, 0.1)

#Pe_rad_range = np.arange(0, 1.1, 0.1)
#Pe_tan_range = np.arange(0, 1.1, 0.1)

start_range = list(np.arange(0, 0.9, 0.01).reshape(9, 10))
start_range.append(np.arange(0.9, 1.01, 0.01))

end_range = list(np.arange(0, 0.9, 0.01).reshape(9, 10))
end_range.append(np.arange(0.9, 1.01, 0.01))

Pe_rad_range = list(np.arange(0, 0.9, 0.01).reshape(9, 10))
Pe_rad_range.append(np.arange(0.9, 1.01, 0.01))

Pe_tan_range = list(np.arange(0, 0.9, 0.01).reshape(9, 10))
Pe_tan_range.append(np.arange(0.9, 1.01, 0.01))
Pe_tan_range = np.concatenate(Pe_tan_range)
Pes_tan = [[Pe_tan, Pe_tan] for Pe_tan in Pe_tan_range]

#Pes_rad = [[Pe_rad, Pe_rad] for Pe_rad in Pe_rad_range]
#Pes_tan = [[Pe_tan, Pe_tan] for Pe_tan in Pe_tan_range]

optimization_net_size = [100, 10, 100]

if __name__=='__main__':

    index = int(sys.argv[1])

    start_index = int(np.floor(index / start_divider))
    end_index = int(np.floor((index - start_index * start_divider) / end_divider))
    rad_index = int(np.floor((index - start_index * start_divider - end_index * end_divider) / rad_divider))

    #start_range = np.array([start_range[start_index]])
    #end_range = np.array([end_range[end_index]])
    #Pes_rad = [Pes_rad[rad_index]]

    start_range = start_range[start_index]
    end_range = end_range[end_index]
    Pe_rad_range = Pe_rad_range[rad_index]
    Pes_rad = [[Pe_rad, Pe_rad] for Pe_rad in Pe_rad_range]

    save_path = params.parameter_optimization_save_path_base + '_' + str(index) + '.pkl'

    parameter_estimation.run_parameter_optimization_iteration(save_path, conduit_element_length=mrad_params.Lce, optimization_net_size=optimization_net_size, 
                                         start_range=start_range, end_range=end_range, Pes_rad=Pes_rad, Pes_tan=Pes_tan)
