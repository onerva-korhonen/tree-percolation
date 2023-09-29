#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:15:20 2023

@author: onerva

This is a pythonization, by Petri Kiuru & Onerva Korhonen, 
of the xylem network model by Mrad et al.

For details of the original model, see

Mrad A, Domec J‐C, Huang C‐W, Lens F, Katul G. A network model links wood anatomy to xylem tissue hydraulic behaviour and vulnerability to cavitation. Plant Cell Environ. 2018;41:2718–2730. https://doi.org/10.1111/pce.13415

Assaad Mrad, Daniel Johnson, David Love, Jean-Christophe Domec. The roles of conduit redundancy and connectivity in xylem hydraulic functions. New Phytologist, Wiley, 2021, pp.1-12. https://doi.org/10.1111/nph.17429

https://github.com/mradassaad/Xylem_Network_Matlab
"""

import numpy as np
import pandas as pd
import openpnm as op
import matplotlib.pyplot as plt

import mrad_params as params

def create_mrad_network(cfg):
    """
    Created a xylem network following the Mrad et al. model

    Parameters
    ----------
    cfg: dic, containing
        save_switch: bln, if True, the network is saved as a numpy npz file
        fixed_random: bln, if True, fixed random seeds are used to create the same network as Mrad's Matlab code
        net_size: np.array, size of the network to be created [TODO: specify how size is defined]
        Lce: float, length of the conduit element (in meters)
        Pc: np.array, propabilities of initiating a conduit at the location closest to the pith (first element)
            and fartest away from it (second element); probabilities for other locations are interpolated from
            these two values
        NPc: np.array, probabilities for terminating an existing conduit, first and second element defined similarly
             as in Pc
        
        

    Returns
    -------
    None.

    """
    save_switch = cfg.get('save_switch',True)
    fixed_random = cfg.get('fixed_random',True)
    net_size = cfg.get('net_size',params.net_size)
    Lce = cfg.get('Lce',params.Lce)
    Pc = cfg.get('Pc',params.Pc)
    NPc = cfg.get('NPc',params.NPc)
    
    if fixed_random:
        seeds_1 = params.seeds_1
        seeds_2 = params.seeds_2
    else:
        # TODO: check what the seeds are used for and if this definition makes sense
        seeds_1 = np.random.randint(0,np.product(net_size),net_size)
        seeds_2 = np.random.randint(0,np.product(net_size),net_size)
    