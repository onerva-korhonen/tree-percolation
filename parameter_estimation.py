#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for optimizing some of the Mrad xylem network model parameters

Created on Tue Nov 12 17:34:18 2024

@author: onervak
"""

import numpy as np
import sys
import os
import pickle
import matplotlib.pylab as plt
from scipy.ndimage import label, generate_binary_structure
from scipy.special import gamma
from operator import itemgetter

import mrad_model
import mrad_params
import params

create_parameter_optimization_data = False
combine_parameter_optimization_data = False
calculate_weibull_params = True
        

def run_parameter_optimization_iteration(save_path, conduit_element_length=mrad_params.Lce, optimization_net_size=[11,10,56], 
                                         start_range=None, end_range=None, Pes_rad=[[1, 1]], Pes_tan=[[1, 1]]):
    """
    Calculates conduit density and length in the space spanned by conduit start and end probability (NPc and Pc) and probability to build radial
    and tangential ICCs (Pe_rad and Pe_tan). Used to create data for optimizing NPc, Pc, Pe_rad, and Pe_tan.
    
    Parameters
    ----------
    save_path : str
        path, to which save the calculated conduit densities and lengths
    conduit_element_length : float, optional
        length of a single conduit element. Default 0.00288 m from the Mrad et al. 2018 article
    optimization_network_size : list if ints, optional
        size of the xylem networks created for the optimization. default [11, 10, 56]
    start_range : np.array of floats, optional
        NPc values to test. default None, in which case values between 0 and 1 are used
    end_range : np.array of floats, optional
        Pc values to test. default None, in which case values between 0 and 1 are used
    Pes_rad : list of iterables of floats, optional
        the probabilities of building an inter-conduit connection in radial direction. Within each element of Pes_rad, the first element corresponds
        to the location closest to the pith and the second element to the location furthest away from it; probabilities for other locations are interpolated from
        these two values (default [[1, 1]])
    Pes_ran : iterable of floats, optional
        the probabilities of building an inter-conduit connection in tangential direction. Within each element of Pes_tan, the first and second elements defined as in
        Pes_rad (default [[1, 1]])
            
        
    Returns
    -------
    None
    """
    assert np.amin(start_range) >= 0, 'probability to start a conduit cannot be negative'
    assert np.amin(end_range) >= 0, 'probability to end a conduit cannot be negative'
    assert min(min(Pes_rad)) >= 0, 'probability to build radial ICCs cannot be negative'
    assert min(min(Pes_tan)) >= 0, 'probability to build tangential ICCs cannot be negative'
    
    cfg = {}
    cfg['net_size'] = optimization_net_size
    cfg['fixed_random'] = False
    
    if start_range is None:
        start_range = np.arange(0, 1.01, 0.01)
    if end_range is None:
        end_range = np.arange(0, 1.01, 0.01)
    conduit_densities = np.zeros((len(start_range), len(end_range), len(Pes_rad), len(Pes_tan)))
    conduit_lengths = np.zeros((len(start_range), len(end_range), len(Pes_rad), len(Pes_tan)))
    grouping_indices = np.zeros((len(start_range), len(end_range), len(Pes_rad), len(Pes_tan)))
    for i, NPc in enumerate(start_range):
        cfg['NPc'] = [NPc, NPc] # assuming same NPc at both ends of the radial range
        for j, Pc in enumerate(end_range): 
            if NPc > 0: # if NPc == 0, there are no conduits and thus conduit density and length equal to zero
                cfg['Pc'] = [Pc, Pc] # assuming same Pc at both ends of the radial range
                for k, Pe_rad in enumerate(Pes_rad):
                    cfg['Pe_rad'] = Pe_rad
                    for l, Pe_tan in enumerate(Pes_tan):
                        cfg['Pe_tan'] = Pe_tan
                        conduit_elements, conns = mrad_model.create_mrad_network(cfg)
                        net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
                        try:
                            net, _ = mrad_model.clean_network(net, conduit_elements, optimization_net_size[0] - 1)
                            conduit_densities[i, j, k, l] += get_conduit_density(net, optimization_net_size)
                            conduit_lengths[i, j, k, l] += get_conduit_length(net)
                            grouping_indices[i, j, k, l] += get_grouping_index(net, optimization_net_size)
                        except Exception as e:
                            if str(e) == 'Cannot delete ALL pores': # current parameter combination doesn't yield any components that would extend through the whole row direction; this can be the case for very small NPc's and very high Pc's
                                conduit_densities[i, j, k, l] = 0
                                conduit_lengths[i, j, k, l] = 0 
                                grouping_indices[i, j, k, l] = 0
                            else:
                                raise

    conduit_lengths *= conduit_element_length
                            
    data = {'start_range' : start_range, 'end_range' : end_range, 'Pes_rad' : Pes_rad, 'Pes_tan' : Pes_tan, 'net_size' : optimization_net_size, 'conduit_densities' : conduit_densities,
            'conduit_lengths' : conduit_lengths, 'grouping_indices' : grouping_indices}
    
    save_folder = save_path.rsplit('/', 1)[0]
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
                    

def optimize_parameters_from_data(target_density, target_length, target_grouping_index, optimization_data_save_folder, optimization_data_save_name_base, fig_save_path_base):
    """
    Starting from pre-calculated data,  finds the conduit start and end probabilities (NPc and Pc) of the Mrad xylem network model that produce
    conduit density, average conduit length, and grouping index as close as possible to given target values.
    
    Parameters
    ----------
    target_density : float
        the desired conduit density
    target_length : float
        the desired average conduit length in meters
    target_grouping_index : float
        the desired grouping index
    optimization_data_save_folder : str
        path of the folder in which the simulation data is saved
    optimization_data_save_name_base : str
        stem of the simulation data file names; the file names can contain other parts but files with names that don't contain this stem aren't read
    fig_save_path : str
        base path, to which save the optimization outcome visualizations
        
    Returns
    -------
    NPc : float
        conduit start probability
    Pc : float
        conduit end probability
    Pe_rad : list of floats
        probability to build an ICC in radial direction, first element corresponding to the location closest to the pith and second element to the location
        furthest away from it
    Pe_tan : list of floats
        probability to build an ICC in tangential direction, first and second element defined as in Pe_rad
    achieved_density : float
        the conduit density produced by NPc and Pc
    achieved_length : float
        the average conduit length produced by Npc and Pc
    """
    data_files = [os.path.join(optimization_data_save_folder, file) for file in os.listdir(optimization_data_save_folder) if os.path.isfile(os.path.join(optimization_data_save_folder, file))]
    data_files = [data_file for data_file in data_files if optimization_data_save_name_base in data_file]

    start_range, end_range, Pes_rad, Pes_tan, conduit_densities, conduit_lengths, grouping_indices = read_and_combine_data(data_files)


    density_landscape = np.argsort(np.abs(conduit_densities - target_density), axis=None).argsort() # rank of absolute difference
    if target_length > 0: # TODO: remove this case after setting target length
        length_landscape = np.argsort(np.abs(conduit_lengths - target_length), axis=None).argsort()
    else:
        length_landscape = np.zeros(conduit_lengths.size)
    grouping_index_landscape = np.argsort(np.abs(grouping_indices - target_grouping_index), axis=None).argsort()
    optimization_landscape = density_landscape + length_landscape + grouping_index_landscape

    optimization_landscape = np.reshape(optimization_landscape, conduit_densities.shape) # the optimal parameter combination minimizes the rank sum of conduit density, conduit length, and grouping index
     
    optimal_NPc_indices, optimal_Pc_indices, optimal_Pe_rad_indices, optimal_Pe_tan_indices = np.where(optimization_landscape == np.amin(optimization_landscape)) 
    #optimal_NPc_indices = [6]
    #optimal_Pc_indices = [6]
    #optimal_Pe_rad_indices = [7]
    #optimal_Pe_tan_indices = [7]
    # TODO: consider adding a tolerance parameter: include in potential optima everything that is within the tolerance from the target
    
    NPc = start_range[optimal_NPc_indices[0]]
    Pc = end_range[optimal_Pc_indices[0]]
    Pe_rad = Pes_rad[optimal_Pe_rad_indices[0]]
    Pe_tan = Pes_tan[optimal_Pe_tan_indices[0]]
    achieved_density = conduit_densities[optimal_NPc_indices[0], optimal_Pc_indices[0], optimal_Pe_rad_indices[0], optimal_Pe_tan_indices[0]]
    achieved_length = conduit_lengths[optimal_NPc_indices[0], optimal_Pc_indices[0], optimal_Pe_rad_indices[0], optimal_Pe_tan_indices[0]]
    achieved_grouping_index = grouping_indices[optimal_NPc_indices[0], optimal_Pc_indices[0], optimal_Pe_rad_indices[0], optimal_Pe_tan_indices[0]]
    
    # Pes_rad and Pes_tan are lists of value pairs; visualization needs lists of single values
    if np.all(np.array([Pe_rad[0] == Pe_rad[1] for Pe_rad in Pes_rad])):
        Pes_rad = np.array([Pe_rad[0] for Pe_rad in Pes_rad]) # if the first and second element are equal, let's use the first elements
    else:
        Pes_rad = np.arange(len(Pes_rad)) # otherwise, let's use the indices of element pairs
    if np.all(np.array([Pe_tan[0] == Pe_tan[1] for Pe_tan in Pes_tan])):
        Pes_tan = np.array([Pe_tan[0] for Pe_tan in Pes_tan])
    else:
        Pes_tan = np.arange(len(Pes_tan))
    
    densities_for_viz_constant_Pe = conduit_densities[:, :, optimal_Pe_rad_indices[0], optimal_Pe_tan_indices[0]]
    densities_for_viz_constant_Pc = conduit_densities[optimal_NPc_indices[0], optimal_Pc_indices[0], :, :]
    lengths_for_viz_constant_Pe = conduit_lengths[:, :, optimal_Pe_rad_indices[0], optimal_Pe_tan_indices[0]]
    lengths_for_viz_constant_Pc = conduit_lengths[optimal_NPc_indices[0], optimal_Pc_indices[0], :, :]
    grouping_indices_for_viz_constant_Pe = grouping_indices[:, :, optimal_Pe_rad_indices[0], optimal_Pe_tan_indices[0]]
    grouping_indices_for_viz_constant_Pc = grouping_indices[optimal_NPc_indices[0], optimal_Pc_indices[0], :, :]
    viz_data = [densities_for_viz_constant_Pe, densities_for_viz_constant_Pc, lengths_for_viz_constant_Pe, lengths_for_viz_constant_Pc, grouping_indices_for_viz_constant_Pe, grouping_indices_for_viz_constant_Pc]
    constant_indices = [[optimal_NPc_indices, optimal_Pc_indices], [optimal_Pe_rad_indices, optimal_Pe_tan_indices], [optimal_NPc_indices, optimal_Pc_indices], [optimal_Pe_rad_indices, optimal_Pe_tan_indices], [optimal_NPc_indices, optimal_Pc_indices], [optimal_Pe_rad_indices, optimal_Pe_tan_indices]]
    x_ranges = [start_range, Pes_rad, start_range, Pes_rad, start_range, Pes_rad]
    y_ranges = [end_range, Pes_tan, end_range, Pes_tan, end_range, Pes_tan]
    z_scales = ['linear', 'linear', 'linear', 'linear', 'log', 'log']
    ylabels = ['NPc', 'Pe_rad', 'NPc', 'Pe_rad', 'NPc', 'Pe_rad']
    xlabels = ['Pc', 'Pe_tan', 'Pc', 'Pe_tan', 'Pc', 'Pe_tan']
    zlabels = ['Conduit density', 'Conduit density', 'Conduit length (m)', 'Conduit length (m)', 'Grouping index', 'Grouping index']
    vmins = [params.conduit_density_vmin, params.conduit_density_vmin, params.conduit_length_vmin, params.conduit_length_vmin, params.grouping_index_vmin, params.grouping_index_vmin]
    vmaxs = [params.conduit_density_vmax, params.conduit_density_vmax, params.conduit_length_vmax, params.conduit_length_vmax, params.grouping_index_vmax, params.grouping_index_vmax]
    save_path_identifiers = ['_conduit_density_constant_Pe', '_conduit_density_constant_Pc', '_conduit_length_constant_Pe', '_conduit_length_constant_Pc', '_grouping_index_constant_Pe', '_grouping_index_constant_Pc']
    
    
    for data, constant_index, x_range, y_range, z_scale, ylabel, xlabel, zlabel, vmin, vmax, save_path_identifier in zip(viz_data, constant_indices, x_ranges, y_ranges, z_scales, ylabels, xlabels, zlabels, vmins, vmaxs, save_path_identifiers):

        centers = [x_range.min(), x_range.max(), y_range.min(), y_range.max()]
        
        dx, = np.diff(centers[:2])/(data.shape[1]-1)
        dy, = -np.diff(centers[2:])/(data.shape[0]-1)
        extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
    
        contours = np.ones(data.shape)
        contours[constant_index[0], constant_index[1]] = 0
    
        contours = np.vstack([contours, np.ones(len(y_range))])
        contours = np.vstack([np.ones(len(y_range)), contours])
        contours = np.hstack([contours, np.ones((len(x_range) + 2, 1))])
        contours = np.hstack([np.ones((len(x_range) + 2, 1)), contours])
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        plt.contour(np.hstack([np.array([np.amin(x_range) - dx]), x_range, np.array([np.amax(x_range) + dx])]), np.hstack([np.array([np.amin(y_range) + dy]), y_range, np.array([np.amax(y_range) - dy])]), contours, 1, colors=params.param_optimization_conduit_color) # NOTE: dx > 0, dy < 0
        plt.imshow(data, origin='lower', extent=extent, norm=z_scale, vmin=vmin, vmax=vmax)
        try:
            plt.colorbar(label=zlabel)
        except:
            continue
    
        ax.set_yticks(x_range)
        ax.set_xticks(y_range)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        save_path = fig_save_path_base + save_path_identifier + '.pdf'
        plt.savefig(save_path, format='pdf',bbox_inches='tight')
    
    return NPc, Pc, Pe_rad, Pe_tan, achieved_density, achieved_length, achieved_grouping_index

def get_conduit_density(net, net_size, conduit_element_length=1.):
    """
    Calculates the conduit density of a given op.Network. The conduit density is obtained by first
    calculating at each row the fraction of cells belonging to a conduit out of all possible cells
    (n_columns x n_depth) and then averaging across the rows.

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements, throats to CECs and ICCs
    net_size : list of ints
        size of the network, [n_rows x n_columns x n_depth]
    conduit_element_length : float, optional
        length of a single conduit element; dividing net['pore.coords'] by conduit_element_length
        should give positive natural numbers. default: 1.

    Returns
    -------
    conduit_density : float
        conduit density of net
    """
    if not 'pore.coords' in net.keys():
        return 0 # there are no pores (= nodes) and thus no conduits in the network
    else:
        pore_coords = net['pore.coords'] / conduit_element_length
        conduit_densities = np.zeros(net_size[0])
        for i in range(net_size[0]):
            conduit_densities[i] = np.sum(pore_coords[:, 0] == i) / (net_size[1] * net_size[2])
        conduit_density = np.mean(conduit_densities)
        return conduit_density

def get_conduit_length(net, cec_indicator=mrad_params.cec_indicator, conduit_element_length=1.):
    """
    Calculates the average conduit length in a given network

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements, throaths to CECs and ICCs
    cec_indicator : int, optional
        value used to indicate that a throat is CEC, default value 1000 (from Mrad et al. 2019)
    conduit_element_length : float, optional
        length of a conduit element. default value 1. yields conduit length measured as the number of conduit elements

    Returns
    -------
    conduit_length : float
        the average conduit length in net
    """
    if not 'throat.conns' in net.keys():
        return 0 # there are no throats (= links) in the network and thus no conduits
    else:
        conns = net['throat.conns']
        conn_types = net['throat.type']
        cec_mask = conn_types == cec_indicator # cec_mask == 1 if the corresponding throat is a connection between two elements in the same conduit
        cecs = conns[cec_mask]
        conduits = mrad_model.get_conduits(cecs)
        conduit_length = conduit_element_length * np.mean(conduits[:, 2])
        return conduit_length
    
def get_grouping_index(net, net_size):
    """
    Calculates the average grouping index, defined as N_conduits / N_conduitgroups, across network slices in row direction. A conduit group
    is defined as a group of conduits touching each other vertically, horizontally or diagonally; solitary conduits count as groups of
    their own.

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements, throats to CECs and ICCs
    net_size : list of ints
        size of the network, [n_rows x n_columns x n_depth]

    Returns
    -------
    grouping_index : float
        average grouping index of net
    """
    if not 'pore.coords' in net.keys(): # there are no pores and thus no conduits and no grouping in the network
        return 0
    else:
        
        pore_coordinates = net['pore.coords']
        n_rows = net_size[0]
        n_columns = net_size[1]
        n_depth = net_size[2]
        grouping_index = 0
        for row_index in np.arange(n_rows):
            occupied_cells = pore_coordinates[np.where(pore_coordinates[:, 0] == row_index)][:, 1:]
            mask = np.zeros((n_columns, n_depth))
            mask[occupied_cells[:, 0], occupied_cells[:, 1]] = 1
            _, num_features = label(mask)
            grouping_index += occupied_cells.shape[0] / num_features
        grouping_index /= n_rows
        return grouping_index
    
def read_and_combine_data(data_files):
    """
    Reads parameter optimization data and combines areas of simulation space possibly distributed in different files into a single space and, when relevant,
    averaging values obtained with the same parameter combination.

    Parameters
    ----------
    data_files : list of strs
        paths to files to be read

    Returns
    -------
    conduit_densities : np.array
        conduit density obtained with different parameter combinations
    conduit_lengths : np.array 
        conduit length obtained with different parameter combinations
    grouping_indices : np.array
        grouping indices obtained with different parameter combinations
    unique_NPcs : np.array
        unique values of NPc used for obtaining data
    unique_Pcs : np.array
        unique Pc values used for obtaining data
    unique_Pes_rad : np.array
        unique Pe_rad values used for obtaining data
    unique_Pes_tan : 
        unique Pe_tan values used for obtaining data
    """
    unique_NPcs = []
    unique_Pcs = []
    unique_Pes_rad = []
    unique_Pes_tan = []
    
    conduit_densities = []
    conduit_lengths = []
    grouping_indices = []
    param_combinations = []

    for i, data_file in enumerate(data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            f.close()
            for NPc in data['start_range']:
                if not NPc in unique_NPcs:
                    unique_NPcs.append(NPc)
            for Pc in data['end_range']:
                if not Pc in unique_Pcs:
                    unique_Pcs.append(Pc)
            for Pe_rad in data['Pes_rad']:
                if not Pe_rad in unique_Pes_rad:
                    unique_Pes_rad.append(Pe_rad)
            for Pe_tan in data['Pes_tan']:
                if not Pe_tan in unique_Pes_tan:
                    unique_Pes_tan.append(Pe_tan)
            conduit_densities.extend(data['conduit_densities'].ravel())
            conduit_lengths.extend(data['conduit_lengths'].ravel())
            grouping_indices.extend(data['grouping_indices'].ravel())
            for Pe_tan in data['Pes_tan']: # TODO: try to make this more efficient
                for Pe_rad in data['Pes_rad']:
                    for Pc in data['end_range']:
                        for NPc in data['start_range']:
                            param_combinations.append(np.array([NPc, Pc, Pe_rad[0], Pe_tan[0]]))
    
    # TODO: the code now assumes the two elements of Pe_rad and Pe_tan to be identical; modify to handle non-identical elements
    unique_NPcs = np.sort(unique_NPcs)
    unique_Pcs = np.sort(unique_Pcs)
    unique_Pes_rad = np.array(sorted(unique_Pes_rad, key=itemgetter(0))) #sorting by the first value
    unique_Pes_tan = np.array(sorted(unique_Pes_tan, key=itemgetter(0)))

    output_param_combinations = []
    for Pe_tan in unique_Pes_tan: # TODO: this looping is rather inefficient; see if anything could be done
        for Pe_rad in unique_Pes_rad:
            for Pc in unique_Pcs:
                for NPc in unique_NPcs:
                    output_param_combinations.append(np.array([NPc, Pc, Pe_rad[0], Pe_tan[0]]))
    output_param_combinations = np.array(output_param_combinations)
    conduit_density_array = np.zeros(len(output_param_combinations))
    conduit_length_array = np.zeros(len(output_param_combinations))
    grouping_index_array = np.zeros(len(output_param_combinations))
    n_iterations = np.zeros(len(output_param_combinations))
    
    for param_combination, conduit_density, conduit_length, grouping_index in zip(param_combinations, conduit_densities, conduit_lengths, grouping_indices):
        index = np.where((output_param_combinations == param_combination).all(axis=1))
        conduit_density_array[index] += conduit_density
        conduit_length_array[index] += conduit_length
        grouping_index_array[index] += grouping_index
        n_iterations[index] += 1

    conduit_density_array /= n_iterations
    conduit_density_array[np.where(conduit_density_array == np.inf)] = np.nan # parameter combinations with no iterations lead to inf values; nan is better for visualization
    conduit_length_array /= n_iterations
    conduit_length_array[np.where(conduit_length_array == np.inf)] = np.nan
    grouping_index_array /= n_iterations
    grouping_index_array[np.where(grouping_index_array == np.inf)] = np.nan
    
    conduit_densities = conduit_density_array.reshape(len(unique_Pes_tan), len(unique_Pes_rad), len(unique_Pcs), len(unique_NPcs)).T
    conduit_lengths = conduit_length_array.reshape(len(unique_Pes_tan), len(unique_Pes_rad), len(unique_Pcs), len(unique_NPcs)).T
    grouping_indices = grouping_index_array.reshape(len(unique_Pes_tan), len(unique_Pes_rad), len(unique_Pcs), len(unique_NPcs)).T

    return unique_NPcs, unique_Pcs, unique_Pes_rad, unique_Pes_tan, conduit_densities, conduit_lengths, grouping_indices
            
    
if __name__=='__main__':
    if create_parameter_optimization_data:
        
        start_range = np.arange(0, 1.1, 0.3)
        end_range = np.arange(0, 1.1, 0.3)
        
        Pe_rad_range = np.arange(0, 1.1, 0.3)
        Pe_tan_range = np.arange(0, 1.1, 0.3)
        
        Pes_rad = [[Pe_rad, Pe_rad] for Pe_rad in Pe_rad_range]
        Pes_tan = [[Pe_tan, Pe_tan] for Pe_tan in Pe_tan_range]
        
        index = int(sys.argv[1])
        save_path = params.parameter_optimization_save_path_base + '_' + str(index) + '.pkl'
        
        run_parameter_optimization_iteration(save_path, conduit_element_length=mrad_params.Lce, optimization_net_size=[11,10,56], 
                                                         start_range=start_range, end_range=end_range, Pes_rad=Pes_rad, Pes_tan=Pes_tan)
    if combine_parameter_optimization_data:
        average_diameter = params.Dc
        average_area = np.pi * (average_diameter/2)**2
        target_density = params.target_conduit_density * average_area # transferring the 1/m^2 conduit density from Lintunen & Kalliokoski 2010 to a fraction of occupied cells
        target_length = 0 # TODO: find average conduit length for Betula pendula
        target_grouping_index = params.target_grouping_index
        
        optimization_data_save_folder = params.parameter_optimization_save_path_base.rsplit('/', 1)[0]
        optimization_data_save_name_base = params.parameter_optimization_save_path_base.rsplit('/', 1)[1]
        
        NPc, Pc, Pe_rad, Pe_tan, achived_density, achieved_length, achieved_grouping_index = optimize_parameters_from_data(target_density, target_length, target_grouping_index, optimization_data_save_folder, 
                                                                                                  optimization_data_save_name_base, params.param_optimization_fig_save_path_base)
        print('Optimal NPc: ' + str(NPc))
        print('Optimal Pc:' + str(Pc))
        print('Optimal Pe_rad:' + str(Pe_rad))
        print('Optimal Pe_tan:' + str(Pe_tan))
        print('Target conduit density ' + str(target_density) + ', achieved conduit density ' + str(achived_density))
        print('Target conduit length ' + str(target_length) + ', achieved conduit length ' + str(achieved_length))
        print('Target grouping index: ' + str(target_grouping_index) + ', achieved grouping index: ' + str(achieved_grouping_index))
        
    if calculate_weibull_params:
        #import pdb; pdb.set_trace()
        
        pore_diameters = params.pore_diameters 
        top_pore_diameters = pore_diameters[np.where(pore_diameters >= np.percentile(pore_diameters, 90))] # following Mrad et al., let's use the 10% largest pore diameters
        cv = np.std(top_pore_diameters) / np.mean(top_pore_diameters) # coefficient of variation
        
        water_surface_tension = mrad_params.water_surface_tension
        bpps = 4*water_surface_tension / pore_diameters # bubble propagation pressures as per equation 6 of Mrad et al. 2018
        a = np.percentile(bpps, 75) # using the 75th percentile of all bubble propagation pressures as per Mrad et al. 2018
        
        b_range = np.arange(1, 100, 0.0001)
        y = np.sqrt((gamma((b_range + 2) / b_range)) / (gamma((b_range + 1) / b_range))**2 - 1)
        b = b_range[np.where(np.abs(y - cv) == np.amin(np.abs(y - cv)))][0]
        
        print('Weibull a: ' + str(a))
        print('Weibull b: ' + str(b))
        
        
        
        
        

    
    
                

