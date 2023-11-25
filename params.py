#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:12:44 2023

@author: onerva

Parameters not related to the Mrad et al. article
"""
import numpy as np

net_size = [np.array([110,100,140])]

# paths for saving
network_save_file = '/home/onerva/projects/hidrogat/output/netpoints'
percolation_plot_save_path = '/home/onerva/projects/hidrogat/output/percolation_3D_conduit_phys.pdf'
nonfunctional_componen_size_save_path = '/home/onerva/projects/hidrogat/output/percolation_3D_conduit_nonfunc_volume_phys.pdf'
ninlet_save_path = '/home/onerva/projects/hidrogat/output/percolation_3D_conduit_ninlet_phys.pdf'

# percolation parameters
percolation_type = 'conduit'
break_nonfunctional_components = False

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
