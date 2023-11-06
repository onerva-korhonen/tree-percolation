#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:12:44 2023

@author: onerva

Parameters not related to the Mrad et al. article
"""
# paths for saving
network_save_file = '/home/onerva/projects/hidrogat/output/netpoints'
percolation_plot_save_path = '/home/onerva/projects/hidrogat/output/percolation_3D_site.pdf'

# visualization parameters
visualize_simulations = False
use_cylindrical_coordinates = True
percolation_outcome_colors = ['r', 'k', 'k']
percolation_outcome_alphas = [1, 1, 0.5]
percolation_outcome_labels = ['effective conductance', 'lcc size', 'func lcc size']
percolation_outcome_axindex = [0, 1, 1]
percolation_outcome_ylabels = ['Effective conductance', 'Component size']
