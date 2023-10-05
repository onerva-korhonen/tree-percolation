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
    Creates a xylem network following the Mrad et al. model

    Parameters
    ----------
    cfg: dic, containing
        save_switch: bln, if True, the network is saved as a numpy npz file
        fixed_random: bln, if True, fixed random seeds are used to create the same network as Mrad's Matlab code
        net_size: np.array, size of the network to be created [TODO: specify how size is defined]
        Lce: float, length of the conduit element (in meters)
        NPc: np.array, propabilities of initiating a conduit at the location closest to the pith (first element)
            and fartest away from it (second element); probabilities for other locations are interpolated from
            these two values
        Pc: np.array, probabilities for terminating an existing conduit, first and second element defined similarly
             as in NPc
        Pe_rad: np.array, probabilities for building an inter-conduit connection (ICC) in the radial direction, 
                first and second element defined similarly as in NPc
        Pe_tan: np.array, probabilities for building an inter-conduit connection (ICC) in the tangential direction, 
                first and second element defined similarly as in NPc
        
        

    Returns
    -------
    None.

    """
    # Reading data
    save_switch = cfg.get('save_switch',True)
    fixed_random = cfg.get('fixed_random',True)
    net_size = cfg.get('net_size',params.net_size)
    Lce = cfg.get('Lce',params.Lce)
    Pc = cfg.get('Pc',params.Pc)
    NPc = cfg.get('NPc',params.NPc)
    Pe_rad = cfg.get('Pe_rad', params.Pe_rad)
    Pe_tan = cfg.get('Pe_ran', params.Pe_tan)
    
    rad_dist = np.ones(net_size)
    
    if fixed_random:
        seeds_NPc = params.seeds_NPc
        seeds_Pc = params.seeds_Pc
    
    # Obtaining random locations for conduit start and end points based on NPc and Pc
    NPc_rad = (rad_dist*NPc[0] + (1 - rad_dist)*NPc[1]) # obtaining NPc values at different radial distances by interpolation 
    # NOTE: because of the definition of rad_dist, this gives the same NPc for each location
    cond_start = []
    for i in range(net_size[1]): # looping in the column direction
        if fixed_random:
            np.random.seed(seeds_NPc[i])
        cond_start.append(np.random.rand(net_size[0] + 100, 1, net_size[2]) < (1 - NPc_rad[i]))
        # TODO: is this a mistake? should it be > (1 - NPc_rad[i])?
    cond_start = np.concatenate(cond_start, axis=1)
    
    Pc_rad = (rad_dist*Pc[0] + (1 - rad_dist)*Pc[1])
    cond_end = []
    for i in range(net_size[1]):
        if fixed_random:
            np.random.seed(seeds_Pc[i])
        cond_end.append(np.random.rand(net_size[0] + 100, 1, net_size[2]) < (1 - Pc_rad[i]))
        # TODO: same here, should < be >?
    cond_end = np.concatenate(cond_end, axis=1)
    
    temp_start = np.zeros(1, net_size[1], net_size[2])
    for j in range(net_size[2]): # looping in the radial (depth) direction
        for i in range(net_size[1]): # looping in the column direction
            # construct a conduit at the first row of this column if there is
            # a 1 among the first 50 entires of the cond_start matrix at this column
            # and the corresponding entries of the cond_end are matrix all 0.
            if (np.where(cond_start[0:50, i, j])[0].size > 0) and (np.where(cond_end[0:50, i, j])[0].size == 0):
                temp_start[0, i, j] = 1 
            # construct a conduit at the first row of this column if the last 
            # 1 among the 50 first entries of the cond_start matrix is at a more
            # advanced postition than the last 1 among the 50 entries of the cond_end matrix
            if (np.where(cond_start[0:50, i, j])[0].size > 0) and (np.where(cond_end[0:50, i, j])[0][-1] > np.where(cond_start[0:50, i, j])[0][-1]):
                temp_start[0, i, j] = 1
    
    # Cleaning up the obtained start and end points
    cond_start = cond_start[50:-50, :, :] # removing the extra elements
    cond_start[0, :, :] = temp_start 
    cond_start[-1, :, :] = 0 # no conduit can start at the last row
    
    cond_end = cond_end[50:-50, : , :]
    cond_end[0, :, :] = 0 # no conduit can end at the first row
    cond_end[-1, :, :] = 1 # all existing conduits must end at the last row
    
    conduit_map = trim_conduit_map(cond_start + cond_end*-1)
    conduit_map = clean_conduit_map(conduit_map)
    
    # constructing the conduit array
    start_and_end_indices = np.where(conduit_map)
    start_and_end_coords = np.array(start_and_end_indices[0], start_and_end_indices[1], start_and_end_indices[2]).T
    start_and_end_coords_sorted = pd.DataFrame(start_and_end_coords, columns = ['A','B','C']).sort_values(by=['C', 'B']).to_numpy()
    
    conduits = []
    conduit_index = 0
    node_index = 0
    
    for i in range(0, len(start_and_end_coords_sorted), 2):
        start_row = start_and_end_coords_sorted[i, 0]
        end_row = start_and_end_coords_sorted[i+1, 0]
        conduit = np.zeros(end_row - start_row + 1, 5)
        conduit[:, 0] = np.linspace(start_row, end_row, end_row - start_row + 1).astype(int)
        conduit[:, 1] = start_and_end_coords_sorted[i, 1]
        conduit[:, 2] = start_and_end_coords_sorted[i, 2]
        conduit[:, 3] = conduit_index
        conduit[:, 4] = node_index + np.linspace(0, end_row - start_row, end_row - start_row + 1).astype(int)
        conduit_index += 1
        node_index += end_row - start_row + 1 
        conduits.append(conduit)
        
    conduits = np.concatenate(conduits)
    # conduit contain one row for each element belonging to a conduit and 5 columns:
    # 1) the row index of the element
    # 2) the column index of the element
    # 3) the radial (depth) index of the element
    # 4) the index of the conduit the element belongs to (from 0 to n_contuits)
    # 5) the index of the element (from 0 to n_conduit_elements)
    
    # finding axial nodes (= pairs of consequtive, in the row direction, elements that belong to the same conduit)
    conx_axi = []
    
    for i in range(1, len(conduits)):
        if (conduits[i - 1, 3] == conduits[i, 3]):
            conx_axi.append(np.array([conduits[i - 1, :], conduits[i, :]]))
        
    
            
    
# Conduit map operations
    
def trim_conduit_map(conduit_map, start_ind=1, end_ind=-1, reset_ind=0):
    """
    Removes from a conduit map end points that are located before the first start
    point and start points that are located after the last end point.
    
    Params:
    -------
    conduit_map: np.array, value of the conduit_map at each element indicates if the element
                 starts a conduit, ends a conduit, or continues the state of the previous element (i.e.
                 continues an existing conduit or a gap between conduits)
    start_ind: int, value that indicates the presence of a conduit start point in an element
    end_ind: int, value that indicates the presence of a conduit end point in an element
    reset_ind: int, value that indicates the lack of conduit start and end points
    
    Returns:
    --------
    trimmed_conduit_map: np.array, conduit_map after trimming
    """
    trimmed_conduit_map = np.zeros(conduit_map.shape)
    for j in range(conduit_map.shape[2]):
        conduit_slice = conduit_map[:, :, j]
        start_rows, start_cols = np.where(conduit_slice == start_ind)
        end_rows, end_cols = np.where(conduit_slice == end_ind)
        for i in range(conduit_map.shape[1]): # looping over the column dimension
        # If there are no conduit starts or conduit ends in column i then the
        # conduit_map should have all zeros in that column. 
            start_rows_in_col = start_rows[start_cols == i]
            end_rows_in_col = end_rows[end_cols == i]
            if (start_rows_in_col.size == 0) or (end_rows_in_col.size == 0):
                conduit_slice[:, i] = reset_ind
                continue # TODO: check if this continue can be removed
            elif (start_rows_in_col.size > 0):
                # Removing all conduit ends (possibly) preceding the first conduit start
                first_conduit_start = start_rows_in_col[0]
                conduit_slice[0:first_conduit_start, i] = reset_ind
                # Removing all conduit starts (possibly) following the last conduit end
                last_conduit_end = end_rows_in_col[-1]
                conduit_slice[last_conduit_end+1:, i] = reset_ind
        trimmed_conduit_map[:, :, j] = conduit_slice
    return trimmed_conduit_map
                
def clean_conduit_map(conduit_map, start_ind=1, end_ind=-1, reset_ind=0):
    """
    Removes from a conduit_map all start points following another start points and
    all end points following another end point.

    Params:
    -------
    conduit_map: np.array, value of the conduit_map at each element indicates if the element
                 starts a conduit, ends a conduit, or continues the state of the previous element (i.e.
                 continues an existing conduit or a gap between conduits)
    start_ind: int, value that indicates the presence of a conduit start point in an element
    end_ind: int, value that indicates the presence of a conduit end point in an element
    reset_ind: int, value that indicates the lack of conduit start and end points
    
    Returns:
    --------
    cleaned_conduit_map: np.array, contains 0:s (no start or end point) and 1:s (start or
                         end point); the first 1 of each column is always a start point,
                         followed by an end point etc.)
    """
    cleaned_conduit_map = np.zeros(conduit_map.shape)
    # TODO: looping changed from Petri's code; check if works
    for k, conduit_map_slice in enumerate(conduit_map):
        for j, conduit_map_column in enumerate(conduit_map_slice):
            start_or_end_indices = np.where(conduit_map_column == start_ind or conduit_map_column == end_ind)[0]
            keep = np.ones(len(conduit_map_column)).astype('bool')
            keep[start_or_end_indices[1:]] = conduit_map_column[start_or_end_indices[1:]]*conduit_map_column[start_or_end_indices[0:-1]] == -1
            cleaned_conduit_map[:, j, k] = np.abs(conduit_map_column * keep)
    return cleaned_conduit_map
            
            