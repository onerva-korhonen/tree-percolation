#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:12:44 2023

@author: onerva

Parameters not related to the Mrad et al. article
"""
import numpy as np
from scipy.stats import norm

# params for network creation
net_size = np.array([110,100,56])
fixed_random = True
if fixed_random:
    #seeds_NPc = [205, 9699, 8324, 2123, 1818, 1834, 3042, 5247, 4319, 2912]
    #seeds_Pc = [6118, 1394, 2921, 3663, 4560, 7851, 1996, 5142, 5924,  464]
    seeds_NPc = [935, 1219,  359, 2762, 1200, 2777, 8909, 1385, 6697, 8561,  826,
                 7712, 4203,  267, 1774, 1368, 2996, 2116, 4868, 5330,  327, 2744,
                 2061, 6479, 2606, 8865, 8599, 7404, 9719, 1753, 8624, 8651, 2650,
                 4127, 4017, 9108, 8706, 1203,  766, 4914,  595, 3888, 5163, 8216,
                 8451, 8241, 5592, 8342, 7300, 1065, 6976, 1424, 6110, 4956, 4937,
                 1998, 6831, 9789, 4065, 5537, 2146, 6444, 8459, 6640, 3215, 9118,
                 5837, 6886,  383, 3314, 9637, 8778, 7258, 5477, 1052, 9746, 2704,
                 2101, 4879, 1509, 7882, 8989, 8636, 7311, 5032, 8631, 9583, 9108,
                 7422, 9303,  310, 3929,  697, 7833, 1775, 3720, 4361, 9135, 1910, 8119]
    seeds_Pc = [718, 5950, 7783, 6453, 1235, 5895, 6293, 6185, 3374, 7569, 3808,
                8740, 3636, 6882, 1195, 9113, 6750, 4770, 6411, 2783, 3708, 6291,
                2549, 8290, 6899, 5159, 5826, 2241, 3769, 6411, 3338, 5676, 3492,
                9614, 8625, 7341, 3696, 5696, 6336, 6429, 7604, 8283, 2950, 7756,
                8211, 5784, 1392, 6915, 5173,  941, 7657, 9827,  906, 3270, 6898,
                3315, 2365, 7373, 9246, 6866, 7497, 9281, 9228, 9776, 8887, 6772,
                3510, 5067, 6262, 9034, 5380, 1026, 5234, 8759, 9386, 6562, 4884,
                1913, 5205, 9889, 6112, 6550, 9656, 8623, 2730, 4029, 7695, 1741,
                5590,  756, 6467, 6268, 6011, 2779, 9903, 7910,  462, 1210, 8945, 1703]
    assert len(seeds_NPc) == net_size[1], 'number of NPc seeds should match the number of network columns'
    assert len(seeds_Pc) == net_size[1], 'number of Pc seeds should match the number of network columns'
    seed_ICC_rad = 63083
    seed_ICC_tan = 73956
    
# dimensions of conduit elements
Dc = 18.18232653752158e-9 # diameter of a conduit element
#Dc_ci = [13.3995938309482, 18.4374185582225] # confidence interval of Dc
Dc_std = 10.76131985221804
Dc_cv = Dc_std / Dc
Dc_alpha = 0.05 # alpha value of Dc_ci
Dc_z = norm.ppf(1 - (1 - Dc_alpha) / 2)
    

# paths for saving
network_save_file = '/home/onervak/projects/hidrogat/output/netpoints'
percolation_plot_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_conduit_si.pdf'
nonfunctional_component_size_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_conduit_si_nonfunc_volume.pdf'
ninlet_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_conduit_si_ninlet.pdf'
prevalence_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_conduit_si_prevalence.pdf'
lcc_in_time_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_conduit_si_lcc_in_time.pdf'
percolation_data_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_conduit_si_data.pkl'

# percolation parameters
percolation_type = 'si'
break_nonfunctional_components = False
si_type = 'stochastic'
si_length = 1000
spreading_probability = 0.1
spreading_threshold = 100 # TODO: set a reasonable value
start_conduits = 'random' # options: 'bottom' (= all conduits with a pore at the first row), 'random' and int
# contact angle and surface tension values are from OpenPNM tutorial (https://openpnm.org/examples/tutorials/09_simulating_invasion.html)
air_contact_angle = 120 # degrees
surface_tension = 0.072 # Newtons/meter
pressure = 100 # number of pressure steps used by the drainage algorithm

# visualization parameters
visualize_simulations = False
use_cylindrical_coordinates = True
percolation_outcome_colors = ['r', 'k', 'k']
percolation_outcome_alphas = [1, 1, 0.5]
percolation_outcome_labels = ['effective conductance', 'lcc size', 'func lcc size']
percolation_outcome_axindex = [0, 1, 1]
percolation_outcome_ylabels = ['Effective conductance', 'Component size']
percolation_nonfunctional_component_size_color = 'b'
percolation_nonfunctional_component_size_label = 'total nonfunctional component volume (m^3)'
percolation_nonfunctional_component_size_alpha = 1
percolation_ninlet_color = 'r'
percolation_noutlet_color = 'b'
percolation_ninlet_label = 'Average n inlets'
percolation_noutlet_label = 'Average n outlets'
percolation_ninlet_alpha = 1
percolation_noutlet_alpha = 1
