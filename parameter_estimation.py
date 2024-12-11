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

import mrad_model
import mrad_params
import params

create_parameter_optimization_data = False
combine_parameter_optimization_data = True
        

def run_parameter_optimization_iteration(save_path, conduit_element_length=mrad_params.Lce, optimization_net_size=[11,10,56], n_iterations=1, 
                                         start_range=None, end_range=None, Pe_rad=[1, 1], Pe_tan=[1, 1]):
    """
    Calculates conduit density and length in the space spanned by conduit start and end probability (NPc and Pc) ranges. Used to create
    data for optimizing NPc and Pc.
    
    Parameters
    ----------
    save_path : str
        path, to which save the calculated conduit densities and lengths
    conduit_element_length : float, optional
        length of a single conduit element. Default 0.00288 m from the Mrad et al. 2018 article
    optimization_network_size : list if ints, optional
        size of the xylem networks created for the optimization. default [11, 10, 56]
    n_iterations : int
        number of xylem networks used for calculating conduit density and length for each NPc-Pc pair
    start_range : np.array of floats, optional
        NPc values to test. default None, in which case values between 0 and 1 are used
    end_range : np.array of floats, optional
        Pc values to test. default None, in which case values between 0 and 1 are used
    Pe_rad : iterable of floats, optional
        the probability of building an inter-conduit connection in radial direction at the location closest to the pith (first element)
        and furthest away from it (second element); probabilities for other locations are interpolated from
        these two values (default [1, 1])
    Pe_ran : iterable of floats, optional
        the probability of building an inter-conduit connection in tangential direction, first and second elements defined as in
        Pe_rad (default [1, 1])
            
        
    Returns
    -------
    None
    """
    assert np.amin(start_range) >= 0, 'probability to start a conduit cannot be negative'
    assert np.amin(end_range) >= 0, 'probability to end a conduit cannot be negative'
    cfg = {}
    cfg['net_size'] = optimization_net_size
    cfg['fixed_random'] = False
    
    # TODO: add a dimension for estimating ICC probability (or otherwise, estimate it outside and give as parameter)
    # TODO: check parallelization: might be better to define the parameter space outside this function, run for one parameter combination at time, save outputs, and do optimization in a separate wrapping function
    
    if start_range is None:
        start_range = np.arange(0, 1.01, 0.01)
    if end_range is None:
        end_range = np.arange(0, 1.01, 0.01)
    conduit_densities = np.zeros((len(start_range), len(end_range)))
    conduit_lengths = np.zeros((len(start_range), len(end_range)))
    for it in range(n_iterations):
        for i, NPc in enumerate(start_range):
            cfg['NPc'] = [NPc, NPc] # assuming same NPc at both ends of the radial range
            for j, Pc in enumerate(end_range):
                if NPc > 0: # if NPc == 0, there are no conduits and thus conduit density and length equal to zero
                    cfg['Pc'] = [Pc, Pc] # assuming same Pc at both ends of the radial range
                    conduit_elements, conns = mrad_model.create_mrad_network(cfg)
                    net = mrad_model.mrad_to_openpnm(conduit_elements, conns)
                    try:
                        net, _ = mrad_model.clean_network(net, conduit_elements, optimization_net_size[0] - 1)
                        conduit_densities[i, j] += get_conduit_density(net, optimization_net_size)
                        conduit_lengths[i, j] += get_conduit_length(net)
                    except Exception as e:
                        if str(e) == 'Cannot delete ALL pores': # current combination of NPc and Pc doesn't yield any components that would extend through the whole row direction; this can be the case for very small NPc's and very high Pc's
                            conduit_densities[i, j] = 0
                            conduit_lengths[i, j] = 0 
                            
    conduit_lengths *= conduit_element_length
                            
    data = {'start_range' : start_range, 'end_range' : end_range, 'Pe_rad' : Pe_rad, 'Pe_tan' : Pe_tan, 'net_size' : optimization_net_size, 'conduit_densities' : conduit_densities,
            'conduit_lengths' : conduit_lengths}
    
    save_folder = save_path.rsplit('/', 1)[0]
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
                    

def optimize_parameters_from_data(target_density, target_length, optimization_data_save_folder, optimization_data_save_name_base, density_fig_save_path, length_fig_save_path):
    """
    Starting from pre-calculated data,  finds the conduit start and end probabilities (NPc and Pc) of the Mrad xylem network model that produce
    conduit density and average conduit length as close as possible to given target values.
    
    Parameters
    ----------
    target_density : float
        the desired conduit density
    target_length : float
        the desired average conduit length in meters
    optimization_data_save_folder : str
        path of the folder in which the simulation data is saved
    optimization_data_save_name_base : str
        stem of the simulation data file names; the file names can contain other parts but files with names that don't contain this stem aren't read
    density_fig_save_path : str
        path, to which save the visualization of the optimization outcome related to conduit density
    length_fig_save_path : str
        path, to which save the visulization of the optimization outcome related to conduit_density
        
    Returns
    -------
    NPc : float
        conduit start probability
    Pc : float
        conduit end probability
    achieved_density : float
        the conduit density produced by NPc and Pc
    achieved_length : float
        the average conduit length produced by Npc and Pc
    """
    #import pdb; pdb.set_trace()
    
    data_files = [os.path.join(optimization_data_save_folder, file) for file in os.listdir(optimization_data_save_folder) if os.path.isfile(os.path.join(optimization_data_save_folder, file))]
    data_files = [data_file for data_file in data_files if optimization_data_save_name_base in data_file]
    
    n_iterations = len(data_files)
    
    for i, data_file in enumerate(data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            f.close()
        if i == 0:
            conduit_densities = data['conduit_densities']
            conduit_lengths = data['conduit_lengths']
            start_range = data['start_range']
            end_range = data['end_range']
        else:
            assert np.all(data['start_range'] == start_range), 'different NPc ranges at different parameter optimization iterations!'
            assert np.all(data['end_range'] == end_range), 'different Pc ranges at different parameter optimization iterations!'
            conduit_densities += data['conduit_densities']
            conduit_lengths += data['conduit_lengths']
    
    
    conduit_densities /= n_iterations
    conduit_lengths /= n_iterations
    
    density_landscape = np.abs(conduit_densities - target_density)
    optimal_NPc_indices, optimal_Pc_indices = np.where(density_landscape == np.amin(density_landscape)) 
    # TODO: consider adding a tolerance parameter: include in potential optima everything that is within the tolerance from the target
    
    if len(optimal_NPc_indices) > 1:
        mask = 1000 * np.ones(len(start_range), len(end_range))
        mask[optimal_NPc_indices, optimal_Pc_indices] = 1
        length_landscape = mask * np.abs(conduit_lengths - target_length)
        optimal_NPc_indices, optimal_Pc_indices = np.where(length_landscape == np.amin(length_landscape))
    
    NPc = start_range[optimal_NPc_indices[0]]
    Pc = end_range[optimal_Pc_indices[0]]
    achieved_density = conduit_densities[optimal_NPc_indices[0], optimal_Pc_indices[0]]
    achieved_length = conduit_lengths[optimal_NPc_indices[0], optimal_Pc_indices[0]]
    
    density_contours = np.ones(conduit_densities.shape)
    density_contours[optimal_NPc_indices, optimal_Pc_indices] = 0
    
    centers = [start_range.min(), start_range.max(), end_range.min(), end_range.max()]

    dx, = np.diff(centers[:2])/(conduit_densities.shape[1]-1)
    dy, = -np.diff(centers[2:])/(conduit_densities.shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
    
    density_fig = plt.figure()
    density_ax = density_fig.add_subplot(111)
    
    plt.imshow(conduit_densities, origin='lower', extent=extent)
    plt.colorbar(label='Conduit density')
    plt.contour(start_range, end_range, density_contours, 1, colors=params.param_optimization_conduit_color)
    
    density_ax.set_yticks(start_range)
    density_ax.set_xticks(end_range)
    density_ax.set_ylabel('NPc')
    density_ax.set_xlabel('Pc')
    
    plt.savefig(density_fig_save_path, format='pdf',bbox_inches='tight')
    
    length_contours = np.ones(conduit_lengths.shape)
    length_contours[optimal_NPc_indices, optimal_Pc_indices] = 0
    
    length_fig = plt.figure()
    length_ax = length_fig.add_subplot(111)
    
    plt.imshow(conduit_lengths, origin='lower', extent=extent)
    plt.colorbar(label='Conduit length (m)')
    plt.contour(start_range, end_range, length_contours, 1, colors=params.param_optimization_conduit_color)
    
    length_ax.set_yticks(start_range)
    length_ax.set_xticks(end_range)
    length_ax.set_ylabel('NPc')
    length_ax.set_xlabel('Pc')
    
    plt.savefig(length_fig_save_path, format='pdf',bbox_inches='tight')
    
    return NPc, Pc, achieved_density, achieved_length

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
    
if __name__=='__main__':
    if create_parameter_optimization_data:
        
        start_range = np.arange(0, 1.1, 0.1)
        end_range = np.arange(0, 1.1, 0.1)
        Pe_rad = [1, 1] # this ensures that no conduits are removed by clean_network()
        Pe_tan = [1, 1]
        
        index = int(sys.argv[1])
        save_path = params.parameter_optimization_save_path_base + '_' + str(index) + '.pkl'
        
        run_parameter_optimization_iteration(save_path, conduit_element_length=mrad_params.Lce, optimization_net_size=[11,10,56], n_iterations=1, 
                                                         start_range=start_range, end_range=end_range, Pe_rad=Pe_rad, Pe_tan=Pe_tan)
    if combine_parameter_optimization_data:
        average_diameter = params.Dc
        average_area = np.pi * average_diameter**2
        target_density = params.target_conduit_density / (1E-3**2 / average_area) # transferring the 1/mm^2 conduit density from Lintunen & Kalliokoski 2010 to a fraction of occupied cells
        target_length = 0 # TODO: find average conduit length for Betula pendula
        
        optimization_data_save_folder = params.parameter_optimization_save_path_base.rsplit('/', 1)[0]
        optimization_data_save_name_base = params.parameter_optimization_save_path_base.rsplit('/', 1)[1]
        
        NPc, Pc, achived_density, achieved_length = optimize_parameters_from_data(target_density, target_length, optimization_data_save_folder, optimization_data_save_name_base,
                                                                                  params.param_optimization_density_fig_save_path, params.param_optimization_length_fig_save_path)

    
    
                

