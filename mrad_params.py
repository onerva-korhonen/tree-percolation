#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:49:37 2023

@author: onerva

Contains the xylem network parameters used in the Mrad et al. publications or their
Matlab code (https://github.com/mradassaad/Xylem_Network_Matlab)
and other default parameters for the network construction
"""
import numpy as np

net_size = np.array([11,10,14])
Lce = 0.00288
Pc = np.array([0.75, 0.75])
NPc = np.array([0.75, 0.75])
Pe_rad = np.array([0.9,0.9])
Pe_tan = np.array([0.02, 0.02])
# the numbers used to seed the random number generation for initiating and
# terminating conduits in the Mrad article
seeds_NPc = [205, 9699, 8324, 2123, 1818, 1834, 3042, 5247, 4319, 2912]
seeds_Pc = [6118, 1394, 2921, 3663, 4560, 7851, 1996, 5142, 5924,  464]
seed_ICC_rad = 63083
seed_ICC_tan = 73956


