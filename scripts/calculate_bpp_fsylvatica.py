#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:44:05 2025

@author: Onerva Korhonen

A script for calculating the bubble propagation pressure for different segments of F. sylvatica (should be easily modifiable for other analyses with multiple segments
                                                                                                 as well).
"""
import params
import fsylvatica_params
import pit_membrane

import numpy as np
import sys

cfg = {}
cfg['conduit_diameters'] = 'lognormal'
cfg['conduit_element_length'] = fsylvatica_params.Lce
cfg['tf'] = fsylvatica_params.tf
cfg['Dp'] = fsylvatica_params.Dp
cfg['n_constrictions'] = fsylvatica_params.n_constrictions
cfg['truncnorm_center'] = fsylvatica_params.truncnorm_center
cfg['truncnorm_std'] = fsylvatica_params.truncnorm_std
cfg['truncnorm_a'] = fsylvatica_params.truncnorm_a
cfg['pore_shape_correction'] = fsylvatica_params.pore_shape_correction
cfg['gas_contact_angle'] = fsylvatica_params.gas_contact_angle
cfg['surface_tension'] = params.surface_tension
cfg['min_radius_type'] = 'max'
cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path

cfg['fixed_random'] = False

if __name__=='__main__':
    
    index = int(sys.argv[1])
    
    cfg['Dc'] = fsylvatica_params.Dc[index]
    cfg['Dc_cv'] = fsylvatica_params.Dc_cv[index]
    cfg['fc'] = fsylvatica_params.fc[index]
    cfg['fpf'] = fsylvatica_params.fpf[index]
    cfg['segment_name'] = fsylvatica_params.segment_names[index]
    
    cfg['bpp_data_path'] = params.bubble_propagation_pressure_data_path.split['.'][0] + '_' + fsylvatica_params.segment_names[index] + '.pkl'
    
    Am_space = np.logspace(np.log10(5E-10), np.log10(7E-8), num=1000)
    
    bpps = pit_membrane.calculate_bpp_for_constriction_pores(cfg, Am_space=Am_space, bpp_save_path=cfg['bpp_data_path'])

