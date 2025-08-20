#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:12:44 2023

@author: onerva

Parameters not related to the Mrad et al. article
"""
import numpy as np
from scipy.stats import norm

import mrad_params

# params for network creation
net_size = np.array([100,100,100])
fixed_random = True
if fixed_random:
    if net_size[1] == 10:
        seeds_NPc = [205, 9699, 8324, 2123, 1818, 1834, 3042, 5247, 4319, 2912]
        seeds_Pc = [6118, 1394, 2921, 3663, 4560, 7851, 1996, 5142, 5924,  464]
    else:
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

# probabilities for network construction
NPc = np.array([0.8, 0.8]) # probability to start a new conduit
Pc = np.array([0.9, 0.9]) # probability to end an existing conduit
Pe_rad = np.array([0.01, 0.01]) # probability of radial connection
Pe_tan = np.array([0.9, 0.9]) # probability of tangential connection
    
# anatomical + physiological parameters
Lce = 379E-6 # m; average conduit element length in Betula pendula branches; from Karimi 2014
Dc = 23.5261E-6 # m; diameter of a conduit element in Betula pendula branches; from Held et al. (in preparation)
Dc_std = 13.1038E-6 # m; standard deviation of conduit element diameter in Betula pendula branches; from Held et al. (in preparation)
# Note: Linunen & Kalliokoski 2010 would have Dc = 18.18232653752158e-6, Dc_std = 10.76131985221804e-6 
#Dc_ci = [13.3995938309482, 18.4374185582225] # confidence interval of Dc
Dc_cv = Dc_std / Dc
Dc_alpha = 0.05 # alpha value of Dc_ci
Dc_z = norm.ppf(1 - (1 - Dc_alpha) / 2)
fc = 0.26 # average contact fraction between two conduits; data from Jansen
Dp = 78.276807125728e-9 # m; average pit membrane pore diameter; from Jansen et al. 2009
tf = 20e-9 # m; microfibril strand thickness; from Kaack et al. 2021
average_pit_membrane_area = 1.096*(1E-3)**2 # m^2; average pit membrane area; from Kaack et al. 2021
fpf = 0.517910375603663 # average pit field fraction between two conduits; from Held et al. (in preparation)
Tm = 205E-9 # pit membrane thickness; from Kaack et al. 2021
pore_diameters = 1E-9 * np.array([13.284, 27.234, 33.805, 63.84, 33.542, 104.234, 52.141, 18.277, 20.634, 19.798, 39.639, 69.396, 85.548, 79.804, 51.699, 21.299, 9.307, 
                  35.219, 14.431, 82.702, 34.647, 64.713, 54.059, 60.852, 50.936, 75.162, 38, 76.067, 107.631, 19.31, 54.895, 68.195, 92.235, 106.55, 
                  110.409, 122.917, 114.778, 8.224, 111.476, 45.735]) # m; pore diameter in Betula pendula, from Jansen et al. 2009
icc_length = mrad_params.icc_length

# params for calculating bubble propagation pressure across pit membrane
# Mrad's method:
weibull_a = 9083441.686765894
weibull_b = 29.345799999996878
# method from Kaack et al. 2021, New Phytologist:
n_constrictions = int(np.floor((Tm*1E9 + 20) / 30)) # calculated following the caption of Table 1 in Kaack et al. 2021 but assuming 10 nm between microfibril strands instead of 20 nm
truncnorm_center = 10.E-9 # m; center value of truncated normal distribution; from Kaack et al. 2021, New Phytologist
truncnorm_std = 7.5E-9 # m, standard deviation of truncated normal distribution; from Kaack et al. 2021, New Phytologist
truncnorm_a = 2.5E-9 # m, beginning of the left truncation; from Kaack et al. 2021, New Phytologist
pore_shape_correction = 0.5 # factor for correcting the assumption of round shape of all pores; from Kaack et al. 2021, New Phytologist
gas_contact_angle = 0 # radians; the contact angle between gas and xylem sap; from Kaack et al. 2021, New Phytologist

target_conduit_density = 630.817129361309/(1e-3)**2 # 1/m^2; conduit density in large branches of Betula pendula; from Lintunen & Kalliokoski 2010
target_grouping_index = 2.47 # grouping index in branches of Betula pendula; from Alber et al., Trees 33, 2019
    

# paths for saving
triton = True
identifier = 'spreading_probability_optimization_medium_net_no_spontaneous_embolism_long_si_updated_conduit_anatomy'
#'parameter_estimation_medium_net_large_space'
#'spreading_probability_optimization_medium_net_no_spontaneous_embolism_long_si'
pooled_optimized_spreading_probability_save_name = 'pooled_optimized_spreading_probability.pkl'
pooled_optimized_spreading_probability_vs_empirical_save_name = 'pooled_optimized_spreading_probability_vs_empirical.pkl'

if triton:
    parameter_optimization_save_path_base = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/param_optimization/' + identifier + '/param_optimization_' + identifier
    spreading_probability_optimization_network_save_path_base = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/optimized_spreading_probability/' + identifier + '_networks/spreading_probability_optimization_network_' + identifier
    network_save_file = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/netpoints'
    percolation_plot_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_3D_' + identifier + '.pdf'
    nonfunctional_component_size_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_3D_' + identifier + '_nonfunc_volume.pdf'
    ninlet_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_3D_' + identifier + '_ninlet.pdf'
    prevalence_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_' + identifier + '_prevalence.pdf'
    lcc_in_time_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_3D_' + identifier + '_lcc_in_time.pdf'
    percolation_data_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_3D_' + identifier + '_data.pkl'
    vc_data_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/vc_data_' + identifier + '.pkl'
    vc_plot_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/vc_' + identifier + '.pdf'
    optimized_spreading_probability_save_path_base = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/optimized_spreading_probability/' + identifier + '/optimized_spreading_probability_' + identifier
    optimized_vc_plot_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/vc_optimized_spreading_probability_' + identifier + '.pdf' 
    optimized_vc_plot_vs_empirical_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/vc_optimized_spreading_probability_vs_empirical_' + identifier + '.pdf' 
    optimized_prevalence_plot_save_path_base = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/prevalence/prevalence_all_pressure_diffs_' + identifier 
    optimized_prevalence_plot_vs_empirical_save_path_base = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/prevalence/prevalence_all_pressure_diffs_vs_empirical_' + identifier
    param_optimization_fig_save_path_base = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/parameter_optimization_' + identifier 
    bubble_propagation_pressure_data_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/bubble_propagation_pressure_data_' + identifier + '.pkl'
    conduit_length_distribution_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/conduit_length_distribution_' + identifier + '.pdf'
    
    single_param_visualization_data_paths = ['/m/cs/scratch/networks/aokorhon/tree-percolation/output/percolation_3D_' + identifier + '_data.pkl']
    optimized_vc_plot_data_save_path_bases = ['/m/cs/scratch/networks/aokorhon/tree-percolation/output/optimized_spreading_probability/' + identifier + '/optimized_spreading_probability_' + identifier]
    empirical_vulnerability_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/optimized_spreading_probability/empirical_vc/empirical_vc_b_pendula_gonzales-munoz_2018'
    degree_distribution_fig_save_path = '/m/cs/scratch/networks/aokorhon/tree-percolation/output/degree_distributions_' + identifier + '.pdf'
else:
    parameter_optimization_save_path_base = '/home/onervak/projects/hidrogat/output/param_optimization/' + identifier + '/param_optimization_' + identifier
    spreading_probability_optimization_network_save_path_base = '/home/onervak/projects/hidrogat/output/optimized_spreading_probability/' + identifier + '_networks/spreading_probability_optimization_network_' + identifier
    network_save_file = '/home/onervak/projects/hidrogat/output/netpoints'
    percolation_plot_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_' + identifier + '.pdf'
    nonfunctional_component_size_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_' + identifier + '_nonfunc_volume.pdf'
    ninlet_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_' + identifier + '_ninlet.pdf'
    prevalence_save_path = '/home/onervak/projects/hidrogat/output/percolation_' + identifier + '_prevalence.pdf'
    lcc_in_time_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_' + identifier + '_lcc_in_time.pdf'
    percolation_data_save_path = '/home/onervak/projects/hidrogat/output/percolation_3D_' + identifier + '_data.pkl'
    vc_data_save_path = '/home/onervak/projects/hidrogat/output/vc_data_' + identifier + '.pkl'
    vc_plot_save_path = '/home/onervak/projects/hidrogat/output/vc_' + identifier + '.pdf'
    optimized_spreading_probability_save_path_base = '/home/onervak/projects/hidrogat/output/optimized_spreading_probability/' + identifier + '/optimized_spreading_probability_' + identifier
    optimized_vc_plot_save_path = '/home/onervak/projects/hidrogat/output/vc_optimized_spreading_probability_' + identifier + '.pdf'
    optimized_vc_plot_vs_empirical_save_path = '/home/onervak/projects/hidrogat/output/vc_optimized_spreading_probability_vs_empirical_' + identifier + '.pdf'
    optimized_prevalence_plot_save_path_base = '/home/onervak/projects/hidrogat/output/prevalence/prevalence_all_pressure_diffs_' + identifier
    optimized_prevalence_plot_vs_empirical_save_path_base = '/home/onervak/projects/hidrogat/output/prevalence/prevalence_all_pressure_diffs_vs_empirical_' + identifier
    param_optimization_fig_save_path_base = '/home/onervak/projects/hidrogat/output/parameter_optimization_' + identifier
    bubble_propagation_pressure_data_path = '/home/onervak/projects/hidrogat/output/bubble_propagation_pressure_data_' + identifier + '.pkl'
    conduit_length_distribution_save_path ='/home/onervak/projects/hidrogat/output/conduit_length_distribution_' + identifier + '.pdf'
    
    single_param_visualization_data_paths = ['/home/onervak/projects/hidrogat/output/percolation_3D_' + identifier + '_data.pkl']
    optimized_vc_plot_data_save_path_bases = ['/home/onervak/projects/hidrogat/output/optimized_spreading_probability/' + identifier + '/optimized_spreading_probability_' + identifier]
    empirical_vulnerability_save_path = '/home/onervak/projects/hidrogat/tree-percolation/output/optimized_spreading_probability/empirical_vc/empirical_vc_b_pendula_gonzales-munoz_2018'
    degree_distribution_fig_save_path = '/home/onervak/projects/hidrogat/tree-percolation/output/degree_distribtions_' + identifier + '.pdf'

# percolation parameters
percolation_type = 'si'
si_type = 'physiological'
bpp_type = 'young-laplace_with_constrictions'
break_nonfunctional_components = False
spontaneous_embolism = False
si_length = 10000
si_tolerance_length = 20
spreading_probability = 0.15100000000000002
start_conduits = 'random_per_component' # options: 'random', 'random_per_component' (= one random seed in all network components), 'none' (= no start conduits set, only spontaneous embolism), int, and 'bottom' (= all conduits with a pore at the first row, allowed only when percolation_type == 'drainage')
surface_tension = 0.025 # Newtons/meter; from Kaack et al. 2021
pressure = 100 # number of pressure steps used by the drainage algorithm
pressure_diff = 3.5e6 # Pa, the difference between water and air (bubble) pressures, delta P in the Mrad et al. article
nCPUs = 5
vulnerability_pressure_range = np.arange(0, 3.5, step=0.25)*1E6
vulnerability_probability_range = np.arange(0.001, 1, step=0.1)
optimization_probability_range = np.arange(0.001, 0.02, step=0.0015)#np.logspace(np.log10(0.0001), np.log10(0.02), 15) # spreading probability range used for optimization
n_iterations = 1 # number of iterations used for optimizing spreading probability

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
percolation_linestyles = ['-', '--']
percolation_labels = ['without spontaneous embolism']
physiological_vc_color = 'r'
physiological_vc_ls = '-'
physiological_vc_alpha = 1
stochastic_vc_color = 'k'
stochastic_vc_ls = '--'
stochastic_vc_alpha = 0.5
optimized_vc_linestyles = ['-']
optimized_vc_labels = ['without spontaneous embolism']
optimized_vc_upper_pressure_limit = 2E6
std_alpha = 0.5
prevalence_linestyles = ['-', '--', '-.'] # for total prevalence, prevalence due to spontaneous embolism, prevalence due to spreading
prevalence_labels = ['total', 'spontaneous', 'spreading']
prevalence_colors = ['b', 'r', 'g']
param_optimization_conduit_color = 'r'
p_50_color = 'b'
p_50_alpha = 1
p_50_line_style = '--'

prevalence_ylims = [-0.3, 1.1]
ninlet_ylims = [-50, 400]
nonfunc_volume_ylims = [-0.1E-8, 0.3E-8]
keff_ylims = [-0.05E-15, 0.2E-15]
lcc_ylims = [-700, 5000] 
conduit_density_vmin = 0.0
conduit_density_vmax = 1.0
conduit_length_vmin = 0.0
conduit_length_vmax = 0.3
grouping_index_vmin = 6.0E-1
grouping_index_vmax = 1.0E2

