#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:15:20 2023

@author: onerva

This is a pythonization, by Petri Kiuru & Onerva Korhonen, 
of the xylem network model by Mrad et al.

For details of the original model, see:
- Mrad A, Domec J‐C, Huang C‐W, Lens F, Katul G. A network model links wood anatomy to xylem tissue hydraulic behaviour and vulnerability to cavitation. Plant Cell Environ. 2018;41:2718–2730. https://doi.org/10.1111/pce.13415
- Assaad Mrad, Daniel Johnson, David Love, Jean-Christophe Domec. The roles of conduit redundancy and connectivity in xylem hydraulic functions. New Phytologist, Wiley, 2021, pp.1-12. https://doi.org/10.1111/nph.17429
- https://github.com/mradassaad/Xylem_Network_Matlab
"""
import numpy as np
import pandas as pd
import openpnm as op
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
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths). Note the directions:
            rows: vertical, columns: radial, depth: tangential
        NPc: np.array, propabilities of initiating a conduit at the location closest to the pith (first element)
            and furthest away from it (second element); probabilities for other locations are interpolated from
            these two values
        Pc: np.array, probabilities for terminating an existing conduit, first and second element defined similarly
             as in NPc
        Pe_rad: np.array, probabilities for building an inter-conduit connection (ICC) in the radial direction, 
                first and second element defined similarly as in NPc
        Pe_tan: np.array, probabilities for building an inter-conduit connection (ICC) in the tangential direction, 
                first and second element defined similarly as in NPc
        cec_indicator: int, value used to indicate that the type of a throat is CEC
        icc_indicator: int, value used to indicate that the type of a throat is ICC
        seeds_NPc : list of ints, random generator seeds used for defining start elements of conduits, 
                    used only if fixed_random == True. len(seeds_NPc) must equal to net_size[1]. (default values
                    produce the same results as the Mrad et al. Matlab code)
        seeds_Pc : list of ints, random generator seeds used for defining end elements of conduits, 
                    used only if fixed_random == True. len(seeds_Pc) must equal to net_size[1]. (default values
                    produce the same results as the Mrad et al. Matlab code)
        seed_ICC_rad : int, random generator seed used for creating radial ICCs, used only if fixed_random == True
                       (default value produce the same results as the Mrad et al. Matlab code)
        seed_ICC_tan : int, random generator seed used for creating tangential ICCs, used only if fixed_random == True
                       (default values produce the same results as the Mrad et al. Matlab code)
        
    Returns
    -------
    conduit_elements : np.array
              has one row for each element belonging to a conduit and 5 columns:
              1) the row index of the element
              2) the column index of the element
              3) the radial (depth) index of the element
              4) the index of the conduit the element belongs to (from 0 to n_conduits)
              5) the index of the element (from 0 to n_conduit_elements - 1)
    conns : np.array
              has one row for each connection between conduit elements and five columns, containing
              1) index of the first conduit element of the connection
              2) index of the second conduit element of the connection
              3) a constant indicating connection (throat) type (in Mrad Matlab implementation, 1000: between conduit elements, 100: ICC)
              4) index of the first conduit of the connection
              5) index of the second conduit of the connection
    """
    # Reading data
    fixed_random = cfg.get('fixed_random', True)
    net_size = cfg.get('net_size', params.net_size)
    n_rows = net_size[0]
    n_columns = net_size[1]
    n_depth =  net_size[2]
    Pc = cfg.get('Pc', params.Pc)
    NPc = cfg.get('NPc', params.NPc)
    Pe_rad = cfg.get('Pe_rad', params.Pe_rad)
    Pe_tan = cfg.get('Pe_ran', params.Pe_tan)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    icc_indicator = cfg.get('icc_indicator', params.icc_indicator)
    
    rad_dist = np.ones(n_columns)
    
    if fixed_random:
        seeds_NPc = cfg.get('seeds_NPc', params.seeds_NPc)
        seeds_Pc = cfg.get('seeds_Pc', params.seeds_Pc)
        seed_ICC_rad = cfg.get('seed_ICC_rad', params.seed_ICC_rad)
        seed_ICC_tan = cfg.get('seed_ICC_tan', params.seed_ICC_tan)
    else:
        seeds_NPc = [None for j in range(n_columns)]
        seeds_Pc = [None for j in range(n_columns)]
    
    # Obtaining random locations for conduit start and end points based on NPc and Pc
    
    # obtaining NPc and Pc values at different radial distances by interpolation 
    # NOTE: because of the definition of rad_dist, this gives the same NPc and Pc for each location
    NPcs_rad = (rad_dist*NPc[0] + (1 - rad_dist)*NPc[1]) 
    Pcs_rad = (rad_dist*Pc[0] + (1 - rad_dist)*Pc[1])

    test_cond_start = []
    test_cond_end = []
    for NPc_rad, Pc_rad, seed_NPc, seed_Pc in zip(NPcs_rad, Pcs_rad, seeds_NPc, seeds_Pc):
        conduit_map_column = create_conduit_map_column(n_rows, n_depth, NPc_rad, Pc_rad, seed_NPc, seed_Pc)
        test_cond_start.append(conduit_map_column[0])
        test_cond_end.append(conduit_map_column[1])
    cond_start = np.concatenate(test_cond_start, axis=1)
    cond_end = np.concatenate(test_cond_end, axis=1)
    
    conduit_map = trim_conduit_map(cond_start + cond_end*-1)
    conduit_map = clean_conduit_map(conduit_map)
    
    # constructing the conduit array
    start_and_end_indices = np.where(conduit_map)
    start_and_end_coords = np.array([start_and_end_indices[0], start_and_end_indices[1], start_and_end_indices[2]]).T
    start_and_end_coords_sorted = start_and_end_coords[np.lexsort((start_and_end_coords[:,1], start_and_end_coords[:,2]))]
    
    conduit_elements = []
    conduit_index = 0
    conduit_element_index = 0

    for i in range(0, len(start_and_end_coords_sorted), 2):
        start_row = start_and_end_coords_sorted[i, 0]
        end_row = start_and_end_coords_sorted[i + 1, 0]
        conduit_element = np.zeros((end_row - start_row + 1, 5), dtype=int)
        conduit_element[:, 0] = np.linspace(start_row, end_row, end_row - start_row + 1)
        conduit_element[:, 1] = start_and_end_coords_sorted[i, 1]
        conduit_element[:, 2] = start_and_end_coords_sorted[i, 2]
        conduit_element[:, 3] = conduit_index
        conduit_element[:, 4] = conduit_element_index + np.linspace(0, end_row - start_row, end_row - start_row + 1)
        conduit_index += 1
        conduit_element_index += end_row - start_row + 1 
        conduit_elements.append(conduit_element)
        
    conduit_elements = np.concatenate(conduit_elements)
    # conduit contain one row for each element belonging to a conduit and 5 columns:
    # 1) the row index of the element
    # 2) the column index of the element
    # 3) the radial (depth) index of the element
    # 4) the index of the conduit the element belongs to (from 0 to n_conduits)
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
    outlet_row_index = n_rows - 1
    
    for i, conduit_element in enumerate(conduit_elements):
        row = conduit_element[0]
        column = conduit_element[1]
        depth = conduit_element[2]
        conduit_index = conduit_element[3]
        node_index = conduit_element[4]
        if ((row == 0) or (row == outlet_row_index)):
            continue # no pit connections in the first and last rows
            
        column_distances = np.abs(conduit_elements[:, 1] - column)[i+1:]
        depth_distances = np.abs(conduit_elements[:, 2] - depth)[i+1:]
        
        # check if there is an adjacent element in radial (= column) or tangential (= depth) direction
        # that belongs to another conduit (adjacent elements in the vertical (=row) direction belong to the same conduit by default). 
        # The maximal number of potential connections in a 3D networks is 8 for each element (sides and corners).
        for conduit_element_2, column_distance, depth_distance in zip(conduit_elements[i + 1:], column_distances, depth_distances):
            row2 = conduit_element_2[0]
            column2 = conduit_element_2[1]
            depth2 = conduit_element_2[2]
            conduit2_index = conduit_element_2[3]
            node2_index = conduit_element_2[4]
            
            if (column_distance > 1) and (depth_distance > 1) and (depth2 != max_depth) and (depth != 0):
                break # there are no potential connections between the current conduit_element_2 and the following instances of conduit_element_2 are even further away (the array is sorted), so let's break the loop
            
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
        np.random.seed(seed_ICC_rad)
    prob_rad = np.random.rand(len(pot_conx_rad), 1)
    if fixed_random:
        np.random.seed(seed_ICC_tan)
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
        ICC_conns[i, 2] = icc_indicator
        ICC_conns[i, 3] = conduit_elements[con[0][4].astype(int), 3]
        ICC_conns[i, 4] = conduit_elements[con[1][4].astype(int), 3]
        
    # ICC_conns has for each ICC one row and five columns, containing
    # 1) index of the first conduit element of the ICC
    # 2) index of the second conduit element of the ICC
    # 3) constant indicating connection (throat) type
    # 4) index of the first conduit of the ICC
    # 5) index of the second conduit of the ICC
    
    CEC_conns = np.zeros((len(conx_axi), 5))
    for i, con in enumerate(conx_axi):
        CEC_conns[i, 0] = con[0][4].astype(int)
        CEC_conns[i, 1] = con[1][4].astype(int)
        CEC_conns[i, 2] = cec_indicator
        
    # The three first columns are defined as in ICC_conns. The last two columns are all zeros and added only for getting
    # matching dimensions
        
    conns = np.concatenate([CEC_conns, ICC_conns])
    conns = conns[np.lexsort((conns[:, 1], conns[:, 0]))]
    
    return conduit_elements, conns
            
def prepare_simulation_network(net, cfg):
    """
    Modifies the properties of an OpenPNM network object to prepare it for Stokes flow and advenction-diffusion simulations.
    
    Parameters:
    -----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between the elements
    cfg : dict
        contains (all default values match the Mrad et al. article):
            use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
            Lce: float, length of a conduit element
            Dc: float, average conduit diameter (m)
            Dc_cv: float, coefficient of variation of conduit diameter
            fc: float, average contact fraction between two conduits
            fpf: float, average pit field fraction between two conduits
            conduit_diameters: np.array of floats, diameters of the conduits OR
                               'lognormal' to draw diameters from a lognormal distribution defined by Dc and Dc_cv OR
                               'inherit_from_net' to use pore diameters of the net object
            cec_indicator: int, value used to indicate that the type of a throat is CE
            tf: float, microfibril strand thickness (m)
            icc_length: float, length of an ICC throat (m)

    Returns
    -------
    sim_net : openpnm.Networks
        network ready for performing simulations

    """
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # Parameter reading and preparation
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', True)
    Lce = cfg.get('Lce', params.Lce)
    Dc = cfg.get('Dc', params.Dc)
    Dc_cv = cfg.get('Dc_cv', params.Dc_cv)
    Dp = cfg.get('Dp', params.Dp)
    fc = cfg.get('fc', params.fc)
    fpf = cfg.get('fpf', params.fpf)
    conduit_diameters = cfg.get('conduit_diameters', params.conduit_diameters)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    tf = cfg.get('tf', params.tf)
    icc_length = cfg.get('icc_length', params.icc_length)
    
    coords = net['pore.coords']
    conns = net['throat.conns']
    conn_types = net['throat.type']
    
    if use_cylindrical_coords:
        pore_coords = mrad_to_cartesian(coords, Lce)
    else:
        pore_coords = Lce * coords
    
    # finding throats that belong to each conduit and conduit and pore diameters
    
    cec_mask = conn_types == cec_indicator # cec_mask == 1 if the corresponding throat is a connection between two elements in the same conduit
    cecs = conns[cec_mask]
    iccs = conns[~cec_mask]
    
    conduits = get_conduits(cecs) # contains the start and end elements and size of each conduit

    conduit_indices = np.zeros(np.shape(pore_coords)[0])
    for i, conduit in enumerate(conduits):
        conduit_indices[conduit[0] : conduit[1] + 1] = i + 1
    conduit_indices = conduit_indices.astype(int) # contains the index of the conduit to which each element belongs, indexing starts from 1
    
    conduit_icc_count = np.zeros(len(conduits)) # contains the number of ICCs per conduit
    for i, conduit in enumerate(conduits):
        mask = (iccs[:, :] >= conduit[0]) & (iccs[:, :] < conduit[1] + 1)
        conduit_icc_count[i] = np.sum(mask)

    diameters_per_conduit, pore_diameters = get_conduit_diameters(net, conduit_diameters, conduits)     
    conduit_areas = (conduits[:, 2] - 1) * Lce * np.pi * diameters_per_conduit # total surface (side) areas of conduits; conduits[:, 2] is the number of elements in a conduit so the conduit length is conduits[:, 2] - 1
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Generating an openpnm network for the simulations
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    sim_net = op.network.Network(coords=pore_coords, conns=conns)
    sim_net['throat.type'] = conn_types
    sim_net.regenerate_models()
    sim_net['pore.diameter'] = pore_diameters
    sim_net['pore.volume'] = 0 # here, pores are 2D horizontal surfaces with zero volumes
    # throat diameters equal to the diameters of the adjacent pores
    sim_net.add_model(propname='throat.max_size', model=op.models.misc.from_neighbor_pores, 
                 mode='min', prop='pore.diameter') 
    sim_net.add_model(propname='throat.diameter', model=op.models.misc.scaled, factor=1., prop='throat.max_size')
    sim_net.add_model(propname='throat.length', model=op.models.geometry.throat_length.spheres_and_cylinders,
                      pore_diameter='pore.diameter', throat_diameter='throat.diameter')
    
    # changing the length of the ICC throats
    sim_net['throat.length'][~cec_mask] = icc_length
    
    sim_net.add_model(propname='throat.surface_area',
                      model=op.models.geometry.throat_surface_area.cylinder,
                      throat_diameter='throat.diameter',
                      throat_length='throat.length')
    sim_net.add_model(propname='throat.volume', 
                      model=op.models.geometry.throat_volume.cylinder,
                      throat_diameter='throat.diameter',
                      throat_length='throat.length')
    sim_net.add_model(propname='throat.area',
                      model=op.models.geometry.throat_cross_sectional_area.cylinder,
                      throat_diameter='throat.diameter')
    sim_net.add_model(propname='throat.diffusive_size_factors', 
                      model=op.models.geometry.diffusive_size_factors.spheres_and_cylinders)
    sim_net.add_model(propname='throat.hydraulic_size_factors', 
                      model=op.models.geometry.hydraulic_size_factors.spheres_and_cylinders)
    sim_net.add_model(propname='pore.effective_volume', model=get_effective_pore_volume)
    sim_net['pore.effective_sidearea'] = 4 * sim_net['pore.effective_volume'] / sim_net['pore.diameter'] #The effective lateral surface area of the pore is calculated from the effective pore volume (A_l = dV/dr for a cylinder)
    sim_net['throat.area_m'] = 0.5 * (sim_net['pore.effective_sidearea'][conns[:, 0]] + sim_net['pore.effective_sidearea'][conns[:, 1]]) * fc * fpf # membrane area calculated from OpenPNM pore geometry
    sim_net['throat.area_m_mrad'] = 0.5 * (conduit_areas[conduit_indices[conns[:, 0]] - 1] / \
        conduit_icc_count[conduit_indices[conns[:, 0]] - 1] + conduit_areas[conduit_indices[conns[:, 1]] - 1] / \
        conduit_icc_count[conduit_indices[conns[:, 1]] - 1]) * fc * fpf # membrane area calculated from the Mrad geometry
    pore_area = (Dp + tf)**2 # area of a single pore (m^2)
    sim_net['throat.npore'] = np.floor(sim_net['throat.area_m_mrad'] / pore_area).astype(int)
    
    return sim_net
    
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
        5) the index of the element (from 0 to n_conduit_elements - 1)
    conns : np.array
        has one row for each connection and five columns, containing
        1) index of the first conduit element of the connection
        2) index of the second conduit element of the connection
        3) constant indicating connection (throat) type (in Mrad Matlab implementation, 1000: between elements within a conduit, 100: ICC)
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
    
def clean_network(net, conduit_elements, outlet_row_index, remove_dead_ends=True):
    """
    Cleans an OpenPNM network object by removing
    conduits that don't have a connection to the inlet (first) or outlet (last) row through
    the cluster to which they belong as well as optionally removing dead-end conduits that have degree 1 and no 
    connection to inlet or outlet. Removal of dead ends is done iteratively: if removing of a dead end creates new
    dead ends, they are removed too.

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
        5) the index of the element (from 0 to n_conduit_elements - 1)
    outlet_row_index : int
        index of the last row of the network (= n_rows - 1)
    remove_dead_ends : bln, optional
        should the conduits that have degree 1 and aren't connected to inlet or outlet
        be removed? (default: True)

    Returns
    -------
    net : openpnm.Network()
        the network object after cleaning
    removed_components : list of lists
        components removed because they are not connected to the inlet or to the outlet; each element corresponds to a 
        removed component and contains the indices of the pores in the removed component
    """    
    _, component_indices, component_sizes = get_components(net)
    sorted_components = []
    sorted_indices = np.argsort(component_sizes)[::-1]
    for sorted_index in sorted_indices:
        sorted_components.append(component_indices[sorted_index])
    
    # Remove components that don't extend through the domain in axial direction
    pores_to_remove = []
    for component in sorted_components:
        in_btm = np.sum(conduit_elements[component, 0] == 0) # number of conduit elements belonging to this component at the inlet row
        in_top = np.sum(conduit_elements[component, 0] == outlet_row_index) # number of conduit elements belonging to this component at the outlet row
        if (in_btm == 0) or (in_top == 0):
            pores_to_remove.append(component)
    if len(pores_to_remove) > 0:
        removed_components = pores_to_remove
        pores_to_remove = np.concatenate(pores_to_remove)
    
        conduit_elements = np.delete(conduit_elements, pores_to_remove, 0)
        op.topotools.trim(net, pores=pores_to_remove)
    else:
        removed_components = []
        
    if remove_dead_ends:
        # Remoce any remaining conduits that are not connected to the inlet or outlet
        # and are connected to only one other conduit
    
        throats_trimmed = net['throat.conns'] # each row of throats_trimmed contain the indices of the start and end conduit elements of a connection
    
        # Tabulate the indices of the conduits connected by each throat (based on the
        # original number of conduits before removing the clusters not connected to the inlet or to the outlet)
    
        throat_conduits = np.zeros((len(throats_trimmed), 2)) # each row of throat_conduits will contain the indices of the start and end conduit of a connection
    
        for i, throat in enumerate(throats_trimmed):
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
        
    return net, removed_components
        
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
    
def read_network(net_path, coord_key='pore.coords', conn_key='throat.conns', type_key='throat.type'):
    """
    Reads a network saved in npz format.
    
    Parameters
    ----------
    net_path : str
        path to which the network has been saved
    coord_key : str
        key under which the pore coordinates have been saved
    conn_key : str
        key under which the throats have been saved
    type_key : str
        key under which the throat types have been saved

    Returns
    -------
    net : openpnm.Network() object
    coords : np.array
        coordinates of the network pores
    conns : np.array
        throats of the network
    conn_types : np.array
        types of the throats; contains int values corresponding to CEs and ICCs
    """
    net = np.load(net_path)
    coords = net[coord_key]
    conns = net[conn_key]
    conn_types = net[type_key]
    
    return net, coords, conns, conn_types
    
        
# Other accessories:
    
def create_conduit_map_column(n_rows, n_depth, NPc_rad, Pc_rad, seed_NPc=None, seed_Pc=None):
    """
    TODO: write documentation

    Parameters
    ----------
    n_rows : TYPE
        DESCRIPTION
    n_depth : TYPE
        DESCRIPTION.
    NPc_rad : TYPE
        DESCRIPTION
    Pc_rad : TYPE
        DESCRIPTION
    seed_NPc : TYPE, optional
        DESCRIPTION. The default is None.
    seed_Pc : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if not seed_Pc is None:
        np.random.seed(seed_NPc)
    cond_start = np.random.rand(n_rows + 100, 1, n_depth) > (1 - NPc_rad)
    
    if not seed_NPc is None:
        np.random.seed(seed_Pc)
    cond_end = np.random.rand(n_rows + 100, 1, n_depth) > (1 - Pc_rad)
    
    temp_start = np.zeros((1, 1, n_depth))
    for j in range(n_depth):
        # construct a conduit at the first row of this column if there is
        # a 1 among the first 50 entires of the cond_start matrix at this column
        # and the corresponding entries of the cond_end are matrix all 0.
        if (np.where(cond_start[0:50, 0, j])[0].size > 0) and (np.where(cond_end[0:50, 0, j])[0].size == 0):
            temp_start[0, 0, j] = 1
        # construct a conduit at the first row of this column if the last 
        # 1 among the 50 first entries of the cond_start matrix is at a more
        # advanced postition than the last 1 among the 50 entries of the cond_end matrix
        if (np.where(cond_start[0:50, 0, j])[0].size > 0) and (np.where(cond_end[0:50, 0, j])[0][-1] < np.where(cond_start[0:50, 0, j])[0][-1]):
            temp_start[0, 0, j] = 1
            
    # Cleaning up the obtained start and end points
    cond_start = cond_start[50:-50, :, :] # removing the extra elements
    cond_start[0, :, :] = temp_start 
    cond_start[-1, :, :] = 0 # no conduit can start at the last row
    
    cond_end = cond_end[50:-50, : , :]
    cond_end[0, :, :] = 0 # no conduit can end at the first row
    cond_end[-1, :, :] = 1 # all existing conduits must end at the last row
    
    return cond_start, cond_end

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
        5) the index of the element (from 0 to n_conduit_elements - 1)
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
               degree > 1, 0 otherwise
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

def mrad_to_cartesian(coords, conduit_element_length=params.Lce, heartwood_d=params.heartwood_d):
    """
    Interpreting the Mrad model coordinates as cylindrical ones, constructs a set of Cartesian
    coordinates that can be used to produce a cylinder-like visualization.

    Parameters
    ----------
    coords : np.array
        the Mrad model coordinates: 
            row corresponds to cylindrical z
            column corresponds to cylindrical r
            depth (= index of radial plane) corresponds to cylindrical theta
    conduit_element_length : float, optional
        length of a conduit element (default value from the Mrad et al. article)
    heartwood_d : float, optional
        diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements)
        (default value from the Mrad et al. article)

    Returns
    -------
    cartesian_coords : np.array
        cartesian coordinates
    """
    coords = coords.astype(int)
    
    r = heartwood_d + coords[:, 1]
    theta_step = (2*np.pi)/(np.max(coords[:, 2]) + 1)
    theta_values = np.arange(0, 2*np.pi, theta_step)
    theta = np.array([theta_values[coords[s, 2]] for s in range(coords.shape[0])])
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = coords[:, 0]
    
    cartesian_coords = conduit_element_length * np.stack((y, x, z), axis=1)
    
    return cartesian_coords

def cartesian_to_mrad(coords, max_depth=params.net_size[2], conduit_element_length=params.Lce, heartwood_d=params.heartwood_d):
    """
    Inverts the transformation made by mrad_to_cartesian function above.
    
    Parameters:
    -----------
    coords : np.array
        n_pores x 3, each column corresponding to one coordinate
    max_depth : int
        maximum value of the third dimension coordinate (depth)
    conduit_element_length : float, optional
        length of a single conduit element (m) (default from the Mrad et al. article)
    heartwood_d : float, optional
        diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements)
        (default value from the Mrad et al. article)
        
    Returns:
    --------
    mrad_coords : np.array
        n_pores x 3
    """
    coords = coords / conduit_element_length
    
    mrad_row = coords[:, 2]
    
    r = np.sqrt(coords[:, 1]**2 + coords[:, 0]**2)
    mrad_col = r - heartwood_d
    
    theta_step = (2 * np.pi) / max_depth
    theta_values = np.arange(0, 2 * np.pi, theta_step)
    theta = np.arctan(coords[:, 0] / coords[:, 1])
    mask = np.zeros(len(theta))
    mask[np.where(coords[:, 1] < 0)] = 1
    mask[np.where((coords[:, 1] > 0) & (coords[:, 0] < 0))] = 2
    theta = np.abs(theta + np.pi * mask)
    mrad_depth = [np.argmin(np.abs(theta_values - theta_value)) for theta_value in theta]
    
    mrad_coords = np.stack((mrad_row, mrad_col, mrad_depth), axis=1)
    mrad_coords = np.round(mrad_coords)
    
    return mrad_coords

def get_conduits(cecs):
    """
    Given the start and end nodes of intra-conduit throats (= throats between elements of the same conduit, CECs),
    constructs a list of conduits.

    Parameters
    ----------
    ces : np.array
        one row per throat, first element gives the index of the start node (conduit element) of the throat and 
        second element the index of the end node

    Returns
    -------
    conduits : np.array
        one row per conduit, includes of the first and last nodes (conduit elements) of the conduit
        and the number of nodes in the conduit
    """            
    conduits = []
    conduit_size = 0
    start_node = cecs[0, 0] # start node of the first CE
    end_node = cecs[0, 1] # end node of the first CE
    for i in range(1, len(cecs)):
        conduit_size += 1
        if cecs[i, 0] - cecs[i - 1, 1] > 0: # start node of the present CE is different from the end node of the previous one = conduit ended
            conduits.append(np.array([start_node, end_node, conduit_size + 1]))
            conduit_size = 0
            start_node = cecs[i, 0]
        end_node = cecs[i, 1]
    conduits.append(np.array([start_node, end_node, conduit_size + 2])) # the last CE is not covered by the loop; its end node ends the last conduit
    conduits = np.asarray(conduits)
    return conduits

def get_conduit_elements(net, use_cylindrical_coords=False, conduit_element_length=params.Lce, 
                         heartwood_d=params.heartwood_d, cec_indicator=params.cec_indicator):
    """
    Constructs the conduit elements array describing a given network.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between them
    use_cylindrical_coords : bln, optional
        have the net['pore.coords'] been defined by interpreting the Mrad model coordinates as cylindrical ones?
    conduit_element_length : float, optional
        length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
    heartwood_d : float, optional
        diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements)
        used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
    cec_indicator : int, optional 
        value used to indicate that the type of a throat is CEC

    Returns
    -------
    conduit_elements : np.array
        has one row for each element belonging to a conduit and 5 columns:
        1) the row index of the element
        2) the column index of the element
        3) the radial (depth) index of the element
        4) the index of the conduit the element belongs to (from 0 to n_conduits - 1)
        5) the index of the element (from 0 to n_conduit_elements - 1)
    """
    pore_coords = net['pore.coords']
    conns = net.get('throat.conns', [])
    conn_types = net.get('throat.type', [])
    
    conduit_elements = np.zeros((len(net['pore.coords']), 5))
    
    if use_cylindrical_coords:
        coords = cartesian_to_mrad(pore_coords, max_depth=np.amax(pore_coords[2]), conduit_element_length=conduit_element_length,
                                   heartwood_d=heartwood_d)
        conduit_elements[:, 0:3] = coords
    else:
        conduit_elements[:, 0:3] = pore_coords
    
    if len(conns) > 0:
        cec_mask = conn_types == cec_indicator # cec_mask == 1 if the corresponding throat is a connection between two elements in the same conduit
        cecs = conns[cec_mask]
        conduits = get_conduits(cecs) # contains the start and end elements and size of each conduit
        conduit_indices = np.zeros(np.shape(pore_coords)[0])
        for i, conduit in enumerate(conduits):
            conduit_indices[conduit[0] : conduit[1] + 1] = i
        conduit_indices = conduit_indices.astype(int) # contains the index of the conduit to which each element belongs, indexing starts from 1
        conduit_elements[:, 3] = conduit_indices
    else:
        conduit_elements[:, 3] = np.arange(conduit_elements.shape[0]) # there are no throats so each element is a conduit of its own
    
    conduit_elements[:, 4] = np.arange(conduit_elements.shape[0])
    
    return conduit_elements
    
def get_conduit_diameters(net, diameter_type, conduits, Dc_cv=params.Dc_cv, Dc=params.Dc,):
    """
    Returns a list of diameters per conduit and per conduit element (pore)

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements and throats to connections between them
    diameter_type : str or iterable
        'lognormal' to draw diameters from a lognormal distribution defined by Dc_cv and Dc OR
        'inherit_from_network' to use pore diameters read from net OR
        an iterable of conduit diameter values (len(diameter_type) must equal to len(conduits))
    conduits : np.array
        contains one row per conduits, columns containing first and last elements and the size of the conduit
    Dc_cv : float, optional
        coefficient of variation of conduit diameter, default value is the one used in the Mrad article
    Dc : float, optional
        average conduit diameter (m), default value is the one used in the Mrad article 

    Returns
    -------
    diameters_per_conduit : np.array
        diameter of each conduit
    pore_diameters : np.array
        diameter of each pore; all pores of a conduit have the same diameter
    """
    if diameter_type == 'inherit_from_net': # pore diameters are read from the network
        pore_diameters = net['pore.diameter']
        diameters_per_conduit = [pore_diameters[conduit[0]] for conduit in conduits] # all pores of a conduit have same diameter; the diameter of first pore is used for defining conduit diameter
    elif diameter_type == 'lognormal':
        Dc_std = Dc_cv*Dc
        Dc_m = np.log(Dc**2 / np.sqrt(Dc_std**2 + Dc**2))
        Dc_s = np.sqrt(np.log(Dc_std**2 / (Dc**2) + 1))
        Dcs = np.random.lognormal(Dc_m, Dc_s, len(conduits)) # diameters of conduits drawn from a lognormal distribution
        diameters_per_conduit = get_sorted_conduit_diameters(conduits, Dcs)
    else:
        diameters_per_conduit = diameter_type
    
    if not diameter_type == 'inherit_from_net':
        pore_diameters = np.zeros(np.shape(net['pore.coords'])[0])
        for conduit, diameter in zip(conduits, diameters_per_conduit):
            pore_diameters[conduit[0] : conduit[1] + 1] = diameter # diameters of the pores (= surfaces between conduit elements), defined as the diameters of the conduits the pores belong to
    
    return diameters_per_conduit, pore_diameters

def get_sorted_conduit_diameters(conduits, diameters):
    """
    Sorts the conduit diameters so that the largest diameters are assigned to the longest
    conduits.

    Parameters
    ----------
    conduits : np.array
        one row per conduit, includes the indices of the first and last nodes (conduit elements) of the conduit
        and the number of nodes in the conduit
    diameters : np.array
        n_conduit diameter values, e.g. drawn from a distribution

    Returns
    -------
    diameters_per_conduit : np.array
        the diameter values sorted in an order that corresponds to the order of the conduits
        array; the longest conduits get the largest diameters.
    """
    sorted_conduits = conduits[conduits[:, 2].argsort(kind='stable')]
    sorted_diameters = np.sort(diameters)
    sorted_conduits = np.concatenate([sorted_conduits, sorted_diameters.reshape(len(sorted_diameters), 1)], axis=1)
    sorted_conduits = sorted_conduits[sorted_conduits[:, 0].argsort(kind='stable')]
    diameters_per_conduit = sorted_conduits[:, 3]
    return diameters_per_conduit

def get_effective_pore_volume(net, throat_volume_key='throat.volume', pore_volume_key='pore.volume', conn_key='throat.conns'):
    """
    Calculates the effective pore volume used in advection-diffusion simulations; effective volume
    is defined as the pore volume + half of the volumes of the adjacent throats.
    

    Parameters
    ----------
    net : openpnm.Network()
    throat_volume_key : str, optional
        key, under which the throat volume information is stored. The default is 'throat.volume'.
    pore_volume_key : str, optional
        key, under which the pore volume information is stored. The default is 'pore.volume'.
    conn_key : str, optional
        key, under which the connection information (= start and end pores of throats) is stored

    Returns
    -------
    effective_pore_volumes : np.array 
        effective volumes of pores
    """
    effective_pore_volumes = net[pore_volume_key].copy().astype(float) # this may contain integer zeros, and apparently in some numpy versions, add.at can't add floats to ints
    throat_volumes = net[throat_volume_key].copy()
    total_volume = effective_pore_volumes.sum() + throat_volumes.sum()
    np.add.at(effective_pore_volumes, net[conn_key][:, 0], net[throat_volume_key]/2)
    np.add.at(effective_pore_volumes, net[conn_key][:, 1], net[throat_volume_key]/2)
    assert np.isclose(effective_pore_volumes.sum(), total_volume), 'total effective pore volume does not match the total volume of pores and throats, please check throat information'
    return effective_pore_volumes

def get_components(net):
    """
    Finds the connected components of an openPNM network and calculates component sizes.

    Parameters
    ----------
    net : openpnm.Network

    Returns
    -------
    component_labels : np.array
        for each pore (node), the index of the component to which the node belongs to
    component_indices : list
        for each component, the indices of nodes belonging to the component
    component_sizes : np.array
        sizes of the components
    """
    try:
        A = net.create_adjacency_matrix(fmt='coo', triu=True) # the rows/columns of A correspond to conduit elements and values indicate the presence/absence of connections
    except KeyError as e:
        if str(e) == "'throat.conns'":
            if len(net['throat.all']) == 0:
                A = np.zeros((len(net['pore.coords']), len(net['pore.coords'])))
            else:
                raise
    component_labels = csg.connected_components(A, directed=False)[1]
    component_indices = []
    if np.unique(component_labels).size > 1:
        for component_label in np.unique(component_labels):
            component_indices.append(np.where(component_labels == component_label)[0])
        component_sizes = np.array([len(component_index) for component_index in component_indices])
    else:
        component_indices = np.array([np.arange(net['pore.coords'].shape[0])])
        component_sizes = [net['pore.coords'].shape[0]]
    return component_labels, component_indices, component_sizes
