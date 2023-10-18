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
import scipy.sparse.csgraph as csg

import mrad_params as params

def create_mrad_network(cfg):
    """
    Creates a xylem network following the Mrad et al. model

    Parameters
    ----------
    cfg : dict
        contains
        fixed_random: bln, if True, fixed random seeds are used to create the same network as Mrad's Matlab code
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
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
    conduit_elements : np.array
              has one row for each element belonging to a conduit and 5 columns:
              1) the row index of the element
              2) the column index of the element
              3) the radial (depth) index of the element
              4) the index of the conduit the element belongs to (from 0 to n_conduits)
              5) the index of the element (from 0 to n_conduit_elements)
    conns : np.array
              has one row for each connection between conduit elements and five columns, containing
              1) index of the first conduit element of the connection
              2) index of the second conduit element of the connection
              3) a constant indicating connection (throat) type (1000: between conduit elements, 100: ICC)
              4) index of the first conduit of the connection
              5) index of the second conduit of the connection
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
    
    rad_dist = np.ones(net_size[1])
    
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
    
    temp_start = np.zeros([1, net_size[1], net_size[2]])
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
            if (np.where(cond_start[0:50, i, j])[0].size > 0) and (np.where(cond_end[0:50, i, j])[0][-1] < np.where(cond_start[0:50, i, j])[0][-1]):
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
    start_and_end_coords = np.array([start_and_end_indices[0], start_and_end_indices[1], start_and_end_indices[2]]).T
    start_and_end_coords_sorted = pd.DataFrame(start_and_end_coords, columns = ['A','B','C']).sort_values(by=['C', 'B']).to_numpy()
    
    conduit_elements = []
    conduit_index = 0
    conduit_element_index = 0
    
    for i in range(0, len(start_and_end_coords_sorted), 2):
        start_row = start_and_end_coords_sorted[i, 0]
        end_row = start_and_end_coords_sorted[i+1, 0]
        conduit_element = np.zeros((end_row - start_row + 1, 5))
        conduit_element[:, 0] = np.linspace(start_row, end_row, end_row - start_row + 1).astype(int)
        conduit_element[:, 1] = start_and_end_coords_sorted[i, 1]
        conduit_element[:, 2] = start_and_end_coords_sorted[i, 2]
        conduit_element[:, 3] = conduit_index
        conduit_element[:, 4] = conduit_element_index + np.linspace(0, end_row - start_row, end_row - start_row + 1).astype(int)
        conduit_index += 1
        conduit_element_index += end_row - start_row + 1 
        conduit_elements.append(conduit_element)
        
    conduit_elements = np.concatenate(conduit_elements)
    # conduit contain one row for each element belonging to a conduit and 5 columns:
    # 1) the row index of the element
    # 2) the column index of the element
    # 3) the radial (depth) index of the element
    # 4) the index of the conduit the element belongs to (from 0 to n_contuits)
    # 5) the index of the element (from 0 to n_conduit_elements)
    
    # finding axial nodes (= pairs of consequtive, in the row direction, elements that belong to the same conduit)
    conx_axi = []
    
    for i in range(1, len(conduit_elements)):
        if (conduit_elements[i - 1, 3] == conduit_elements[i, 3]):
            conx_axi.append(np.array([conduit_elements[i - 1, :], conduit_elements[i, :]]))
        
    # finding potential pit connections between conduit elements
    
    max_depth = int(np.max(conduit_elements[:, 2]))
    pot_conx_rad = []
    pot_conx_tan = []
    outlet_row_index = net_size[0] - 1
    
    for i, conduit_element in enumerate(conduit_elements):
        row = conduit_element[0]
        column = conduit_element[1]
        depth = conduit_element[2]
        conduit_index = conduit_element[3]
        node_index = conduit_element[4]
        if ((row == 0) or (row == outlet_row_index)):
            continue # no pit connections in the first and last rows
        
        # check if there is a horizontally (= in column direction) adjacent
        # element that is part of another conduit. The maximal number of 
        # potential connections in a 3D networks is 8 for each element.
        for conduit_element_2 in conduit_elements[i + 1:]:
            row2 = conduit_element_2[0]
            column2 = conduit_element_2[1]
            depth2 = conduit_element_2[2]
            conduit2_index = conduit_element_2[3]
            node2_index = conduit_element_2[4]
            
            if ((abs(column2 - column) > 1) and (abs(depth2 - depth) > 1) and (depth2 != max_depth) and (depth != 0)):
                break # the conduit elements in next rows are further away than this one, so let's break the loop
            
            if (row2 == row):
                if (column2 - column == 1) and (depth2 == depth):
                    pot_conx_rad.append(np.array([[row, column, depth, conduit_index, node_index],
                                                 [row2, column2, depth2, conduit2_index, node2_index]]))
                elif (((column2 - column == 1) and (depth2 - depth == 1)) or \
                     ((depth2 - depth == 1) and (column2 - column <= 1) and (column2 - column >= 0)) or \
                     ((depth == 0) and (depth2 == max_depth) and (column2 - column <= 1) and (column2 - column >= 0))):  
                    pot_conx_tan.append(np.array([[row, column, depth, conduit_index, node_index],
                                                 [row2, column2, depth2, conduit2_index, node2_index]]))
                    
    # picking the actual pit connections

    Pe_rad_rad = (rad_dist*Pe_rad[0] + (1 - rad_dist)*Pe_rad[1])
    Pe_tan_rad = (rad_dist*Pe_tan[0] + (1 - rad_dist)*Pe_tan[1])
    
    if fixed_random:
        np.random.seed(params.seed_ICC_rad)
    prob_rad = np.random.rand(len(pot_conx_rad), 1)
    if fixed_random:
        np.random.seed(params.seed_ICC_tan)
    prob_tan = np.random.rand(len(pot_conx_tan), 1)
    
    conx = []
    
    for pot_con, p in zip(pot_conx_rad, prob_rad):
        if (p >= (1 - np.mean(Pe_rad_rad[pot_con[:,1].astype(int)]))):
            conx.append(pot_con)
            
    for pot_con, p in zip(pot_conx_tan, prob_tan):
        if (p >= (1 - np.mean(Pe_tan_rad[pot_con[:,1].astype(int)]))):
            conx.append(pot_con)
                
    ICC_conns = np.zeros((len(conx), 5))
    for i, con in enumerate(conx):
        ICC_conns[i, 0] = con[0][4].astype(int)
        ICC_conns[i, 1] = con[1][4].astype(int)
        ICC_conns[i, 2] = 100
        ICC_conns[i, 3] = conduit_elements[con[0][4].astype(int), 3]
        ICC_conns[i, 4] = conduit_elements[con[1][4].astype(int), 3]
        
    # ICC_conns has for each ICC one row and five columns, containing
    # 1) index of the first conduit element of the ICC
    # 2) index of the second conduit element of the ICC
    # 3) constant indicating connection (throat) type (100: ICC)
    # 4) index of the first conduit of the ICC
    # 5) index of the second conduit of the ICC
    
    CEC_conns = np.zeros((len(conx_axi), 5))
    for i, con in enumerate(conx_axi):
        CEC_conns[i, 0] = con[0][4].astype(int)
        CEC_conns[i, 1] = con[1][4].astype(int)
        CEC_conns[i, 2] = 1000
        
    # The constant in the third column of CEC_conns, 1000, indicates that the connection is between elements in
    # the same conduit. The last two columns are all zeros and added only for getting
    # matching dimensions
        
    conns = np.concatenate([CEC_conns, ICC_conns])
    conns = pd.DataFrame(conns, columns = ['A','B','C','D','E']).sort_values(by=['A', 'B']).to_numpy()
    
    return conduit_elements, conns
            
    
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
                continue
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
    conduit_map = np.transpose(conduit_map)
    cleaned_conduit_map = np.zeros(conduit_map.shape)
    for k, conduit_map_slice in enumerate(conduit_map):
        for j, conduit_map_column in enumerate(conduit_map_slice):
            start_or_end_indices = np.nonzero(conduit_map_column)[0]
            keep = np.ones(len(conduit_map_column)).astype('bool')
            keep[start_or_end_indices[1:]] = conduit_map_column[start_or_end_indices[1:]]*conduit_map_column[start_or_end_indices[0:-1]] == -1
            cleaned_conduit_map[k, j, :] = np.abs(conduit_map_column * keep)
    cleaned_conduit_map = np.transpose(cleaned_conduit_map)
    return cleaned_conduit_map

# OpenPNM-related accessories

def mrad_to_openpnm(conduit_elements, conns):
    """
    Transforms a network created following the Mrad model into an OpenPNM network.

    Parameters
    ----------
    conduit_elements : np.array
        has one row for each element belonging to a conduit and 5 columns:
        1) the row index of the element
        2) the column index of the element
        3) the radial (depth) index of the element
        4) the index of the conduit the element belongs to (from 0 to n_conduits)
        5) the index of the element (from 0 to n_conduit_elements)
    conns : np.array
        has one row for each connection and five columns, containing
        1) index of the first conduit element of the connection
        2) index of the second conduit element of the connection
        3) constant indicating connection (throat) type (1000: between elements within a conduit, 100: ICC)
        4) index of the first conduit of the connection
        5) index of the second conduit of the connection

    Returns
    -------
    pn: openpnm.Network object
        the nodes (pores) of pn correspond to conduit elements of the Mrad model and links (throats)
        the connections between the elements
    """
    ws = op.Workspace()
    ws.clear()
    
    proj = ws.new_project('name = diffpornetwork')
    
    pn = op.network.Network(project=proj, coords=conduit_elements[:, 0:3], conns=conns[:, 0:2].astype(int))
    pn['throat.type'] = conns[:, 2].astype(int)
    
    return pn
    
def clean_network(net, conduit_elements, outlet_row_index):
    """
    Cleans an OpenPNM network object by removing
    conduits that don't have a connection to the inlet (first) or outlet (last) row through
    the cluster to which they belong

    Parameters
    ----------
    net : openpnm.Network()
        network object
    conduit_elements : np.array
        has one row for each element belonging to a conduit and 5 columns:
        1) the row index of the element
        2) the column index of the element
        3) the radial (depth) index of the element
        4) the index of the conduit the element belongs to (from 0 to n_conduits)
        5) the index of the element (from 0 to n_conduit_elements)
    outlet_row_index : int
        index of the last row of the network (= n_rows - 1)

    Returns
    -------
    net : openpnm.Network()
        the network object after cleaning
    """    
    A = net.create_adjacency_matrix(fmt='coo', triu=True) # the rows/columns of A correspond to conduit elements and values indicate the presence/absence of connections
    components = csg.connected_components(A, directed=False)[1]
    component_indices = []
    sorted_components = []
    if np.unique(components).size > 1:
        for component in np.unique(components):
            component_indices.append(np.where(components == component)[0])
        component_sizes = np.array([len(component_index) for component_index in component_indices])
        sorted_indices = np.argsort(component_sizes)[::-1]
        for sorted_index in sorted_indices:
            sorted_components.append(component_indices[sorted_index])
    
    # Remove components that don't extend through the domain in axial direction
    components_to_remove = []
    for component in sorted_components:
        in_btm = np.sum(conduit_elements[component, 0] == 0) # number of conduit elements belonging to this component at the inlet row
        in_top = np.sum(conduit_elements[component, 0] == outlet_row_index) # number of conduit elements belonging to this component at the outlet row
        if (in_btm == 0) or (in_top == 0):
            components_to_remove.append(component)
    components_to_remove = np.concatenate(components_to_remove)
    
    conduit_elements = np.delete(conduit_elements, components_to_remove, 0)
    op.topotools.trim(net, pores=components_to_remove)
    
    # Remoce any remaining conduits that are not connected to the inlet or outlet
    # and are connected to only one other conduit
    
    throats_trimmed = net['throat.conns'] # each row of throats_trimmed contain the indices of the start and end conduit elements of a connection
    
    # Tabulate the indices of the conduits connected by each throat (based on the
    # original number of conduits before removing the clusters not connected to the inlet or to the outlet)
    
    throat_type = np.zeros(len(throats_trimmed))
    throat_conduits = np.zeros((len(throats_trimmed), 2)) # each row of throat_conduits will contain the indices of the start and end conduit of a connection
    
    for i, throat in enumerate(throats_trimmed):
        if conduit_elements[throat[0], 3] == conduit_elements[throat[1], 3]:
            throat_type = 1 # throat inside a conduit
        else:
            throat_type = 2 # ICC
        throat_conduits[i, 0] = conduit_elements[throat[0], 3]
        throat_conduits[i, 1] = conduit_elements[throat[1], 3]
        
    conduit_indices = np.unique(throat_conduits)
    conduit_degree_info = get_conduit_degree(conduit_elements, throat_conduits, outlet_row_index) 
    
    conduits_to_remove = conduit_indices[np.where(conduit_degree_info[:, 3] == 1)]
    while (conduits_to_remove.shape[0] > 0):
        conduit_elements_to_remove = []
        throats_to_remove = []
        for conduit_to_remove in conduits_to_remove:
            conduit_elements_to_remove.append(np.where(conduit_elements[:, 3] == conduit_to_remove)[0])
            throats_to_remove.append(np.where(throat_conduits[:, 0] == conduit_to_remove)[0])
            throats_to_remove.append(np.where(throat_conduits[:, 1] == conduit_to_remove)[0])
        conduit_elements_to_remove = np.concatenate(conduit_elements_to_remove)
        throats_to_remove = np.unique(np.concatenate(throats_to_remove))
        
        for conduit_to_remove in conduits_to_remove:
            conduit_indices = np.delete(conduit_indices, np.where(conduit_indices == conduit_to_remove))
            
        conduit_elements = np.delete(conduit_elements, conduit_elements_to_remove, 0)
        throat_conduits = np.delete(throat_conduits, throats_to_remove, 0)
        op.topotools.trim(network=net, pores=conduit_elements_to_remove)
        
        conduit_degree_info = get_conduit_degree(conduit_elements, throat_conduits, outlet_row_index)
        conduits_to_remove = conduit_indices[np.where(conduit_degree_info[:, 3] == 1)]
        
    return net
        
def save_network(net, save_path):
    """
    Saves a network to a given path as a numpy npz file.

    Parameters
    ----------
    net : openpmn.Network() object
        the pores correspond to conduit elements and throats to connections between the elements
    save_path : str
        path to which save the network

    Returns
    -------
    None.
    """
    np.savez(save_path, coord_cleaned=net['pore.coords'], conns_cleaned=net['throat.conns'], conn_types=net['throat.type'])
    
        
# Other asseccories:

def get_conduit_degree(conduit_elements, throat_conduits, outlet_row_index):
    """
    Calculates the degree (= number of connections to other conduits) 
    of each conduit, the min and max rows the conduit spans, and a boolean
    variable telling if the conduit touches either the inlet (0) or outlet (max)
    row or has degree > 1.

    Parameters
    ----------
    conduit_elements : np.array
        has one row for each element belonging to a conduit and 5 columns:
        1) the row index of the element
        2) the column index of the element
        3) the radial (depth) index of the element
        4) the index of the conduit the element belongs to (from 0 to n_conduits)
        5) the index of the element (from 0 to n_conduit_elements)
    throat_conduits : np.array
        each row contains the indices of the start and end conduits of a throat
    outlet_row_index : int
        index of the last row of the network (= n_rows - 1)
    Returns
    -------
    conduit_degree_info: np.array
        contains one row for each conduit and 4 columns:
            1) degree of the conduit
            2) the smallest row index among the elements of the conduit
            3) the largest row index among the elements of the conduit
            4) 1 if the conduit does not touch either inlet or outlet and has
               degree < 0, 0 otherwise
    """    
    conduit_indices = np.unique(throat_conduits)
    conduit_degree_info = np.zeros((len(conduit_indices), 4))
    for i, conduit_index in enumerate(conduit_indices):
        throat_start_indices = np.where(throat_conduits[:, 0] == conduit_index)
        throat_end_indices = np.where(throat_conduits[:, 1] == conduit_index)
        
        degree = len(np.unique(np.concatenate([throat_conduits[throat_end_indices, 0], 
                                               throat_conduits[throat_start_indices, 1]], axis=1))) - 1
        # degree is calculated by finding the conduits starting a throat ending at the present conduit
        # or ending a throat starting from the present conduit, calculating the number of unique
        # neighbors. If there are any throats between two elements of the present conduit (almost always),
        # the conduit itself is listed as a neighbor. Subtracting 1 compensates for this.
        min_row = np.min(conduit_elements[conduit_elements[:, 3] == conduit_index, 0])
        max_row = np.max(conduit_elements[conduit_elements[:, 3] == conduit_index, 0])
        
        conduit_degree_info[i, 0] = degree
        conduit_degree_info[i, 1] = min_row
        conduit_degree_info[i, 2] = max_row
        if (degree == 1) and ((min_row != 0) and (max_row != outlet_row_index)):
            conduit_degree_info[i, 3] = 1
    
    return conduit_degree_info
            
# Visualization

def visualize_network_with_openpnm(net):
    """
    Visualizes a conduit network using the OpenPNM visualization tools.
    
    Parameters:
    -----------
    net : openpnm.Network() object
        pores correspond to conduit elements, throats to connections between elements

    Returns
    -------
    None.

    """
    ts = net.find_neighbor_throats(pores=net.pores('all'))
    
    ax = op.visualization.plot_coordinates(network=net, pores=net.pores('all'), c='r')
    ax = op.visualization.plot_connections(network=net, throats=ts, ax=ax, c='b')
                                           