#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for optimizing some of the Mrad xylem network model parameters

Created on Tue Nov 12 17:34:18 2024

@author: onervak
"""

import numpy as np

import mrad_model
import mrad_params
import params
        

def optimize_conduit_start_and_end_probabilities(target_density, target_length, conduit_element_length=mrad_params.Lce, optimization_net_size=[11,10,56], n_iterations=1, 
                                                 start_range=[], end_range=[], Pe_rad=[1, 1], Pe_tan=[1, 1]):
    """
    Finds the conduit start and end probabilities (NPc and Pc) of the Mrad xylem network model that produce
    conduit density and average conduit length as close as possible to given target values.
    
    Parameters
    ----------
    target_density : float
        the desired conduit density
    target_length : float
        the desired average conduit length in meters
    conduit_element_length : float, optional
        length of a single conduit element. Default 0.00288 m from the Mrad et al. 2018 article
    optimization_network_size : list if ints, optional
        size of the xylem networks created for the optimization. default [11, 10, 56]
    n_iterations : int
        number of xylem networks used for calculating conduit density and length for each NPc-Pc pair
    start_range : list of floats, optional
        NPc values to test. default [], in which case values between 0 and 1 are used
    end_range : list of floats, optional
        Pc values to test. default [], in which case values between 0 and 1 are used
    Pe_rad : iterable of floats, optional
        the probability of building an inter-conduit connection in radial direction at the location closest to the pith (first element)
        and furthest away from it (second element); probabilities for other locations are interpolated from
        these two values (default [1, 1])
    Pe_ran : iterable of floats, optional
        the probability of building an inter-conduit connection in tangential direction, first and second elements defined as in
        Pe_rad (default [1, 1])
            
        
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
    assert np.amin(start_range) >= 0, 'probability to start a conduit cannot be negative'
    assert np.amin(end_range) >= 0, 'probability to end a conduit cannot be negative'
    cfg = {}
    cfg['net_size'] = optimization_net_size
    cfg['fixed_random'] = False
    
    # TODO: add a dimension for estimating ICC probability (or otherwise, estimate it outside and give as parameter)
    # TODO: check parallelization: might be better to define the parameter space outside this function, run for one parameter combination at time, save outputs, and do optimization in a separate wrapping function
    
    if len(start_range) == 0:
        start_range = np.arange(0, 1.01, 0.01)
    if len(end_range) == 0:
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
                    
    conduit_densities /= n_iterations
    conduit_lengths /= n_iterations
    conduit_lengths *= conduit_element_length
    
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
    average_diameter = params.Dc
    average_area = np.pi * average_diameter**2
    target_density = params.target_conduit_density / (1E-3**2 / average_area) # transferring the 1/mm^2 conduit density from Lintunen & Kalliokoski 2010 to a fraction of occupied cells
    target_length = 0 # TODO: find average conduit length for Betula pendula
    
    start_range = np.arange(0, 1.1, 0.1)
    end_range = np.arange(0, 1.1, 0.1)
    Pe_rad = [1, 1] # this ensures that no conduits are removed by clean_network()
    Pe_tan = [1, 1]
    
    NPc, Pc, _, _ = optimize_conduit_start_and_end_probabilities(target_density, target_length, conduit_element_length=mrad_params.Lce, optimization_net_size=[11,10,56], n_iterations=1, 
                                                     start_range=start_range, end_range=end_range, Pe_rad=Pe_rad, Pe_tan=Pe_tan)
    
    
                

