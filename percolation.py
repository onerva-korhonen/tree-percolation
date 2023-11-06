#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:30:10 2023

@author: onerva

Functions for percolation analysis of the xylem network and related effective conductance
"""
import numpy as np
import networkx as nx
import openpnm as op
import scipy.sparse.csgraph as csg

import mrad_model
import mrad_params as params

def run_percolation(net, cfg, percolation_type='bond', removal_order='random'):
    """
    Removes throats (bond percoltion) or pores (site percolation) from an OpenPNM network object in a given (or random) order and calculates the effective conductance
    and largest connected component size after each removal.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between the elements
    cfg : dict
        contains:
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
        Lce: float, length of a conduit element
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits
        conduit_diameters: np.array of floats, diameters of the conduits, or 'lognormal'
        to draw diameters from a lognormal distribution defined by Dc and Dc_cv
        cec_indicator: int, value used to indicate that the type of a throat is CE
        tf: float, microfibril strand thickness (m)
        icc_length: float, length of an ICC throat (m)
        water_pore_viscosity: float, value of the water viscosity in pores
        water_throat_viscosity: float, value of the water viscosity in throats
        water_pore_diffusivity: float, value of the water diffusivity in pores
        inlet_pressure: float, pressure at the inlet conduit elements (Pa)
        outlet_pressure: float, pressure at the outlet conduit elements (Pa) 
        Dp: float, average pit membrane pore diameter (m)
        Tm: float, average thickness of membranes (m)
        cec_indicator: int, value used to indicate that the type of a throat is CE
    percolation type : str, optional
        type of the percolation, 'bond' to remove throats or 'site' to remove pores (default: 'bond')
    removal_order : iterable or str, optional (default='random')
        indices of pores to be removed in the order of removal. for performing full percolation analysis, len(removal_order)
        should match the number of pores. string 'random' can be given to indicate removal in random order.

    Returns
    -------
    effective_conductances : np.array
        effective conductance of the network after each removal
    lcc_size : np.array
        size of the largest connected component after each removal
    functional_lcc_size : np.array
        the size of the largest functional component (i.e. component connected to both inlet and outlet) after each removal
    nonfunctional_component_size : np.array
        total size (i.e. sum of sizes) of non-functional components (i.e. components not connected to either inlet, outlet, or both) after each removal
    susceptibility : np.array
        susceptibility (i.e. the second-largest component size) after each removal
    functional_susceptibility : np.array
        functional susceptibility (i.e. the size of the second-largest component connected to inlet and outlet) after each removal
    """
    assert percolation_type in ['bond', 'site'], 'percolation type must be bond (removal of throats) or site (removal of pores)'
    if percolation_type == 'bond':
        n_removals = net['throat.conns'].shape[0]
    elif percolation_type == 'site':
        n_removals = net['pore.coords'].shape[0]
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    
    effective_conductances = np.zeros(n_removals)
    lcc_size = np.zeros(n_removals)
    functional_lcc_size = np.zeros(n_removals)
    nonfunctional_component_size = np.zeros(n_removals)
    susceptibility = np.zeros(n_removals)
    functional_susceptibility = np.zeros(n_removals)
    cfg['conduit_diameters'] = 'inherit_from_net'
    if removal_order == 'random':
        removal_order = np.arange(0, n_removals)
        np.random.shuffle(removal_order)
    for i, _ in enumerate(removal_order):
        sim_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
        sim_net['throat.type'] = net['throat.type']
        if 'pore.diameter' in net.keys():
            sim_net['pore.diameter'] = net['pore.diameter']
        if percolation_type == 'bond':
            op.topotools.trim(sim_net, throats=removal_order[:i + 1])
        elif percolation_type == 'site':
            try:
                op.topotools.trim(sim_net, pores=removal_order[:i + 1])
            except Exception as e:
                if str(e) == 'Cannot delete ALL pores':
                    nonfunctional_component_size[i] = len(net['pore.coords'])
                    continue # this would remove the last node from the network; at this point, value of all outputs should be 0
        lcc_size[i], susceptibility[i] = get_lcc_size(sim_net)
        try:
            conduit_elements = mrad_model.get_conduit_elements(sim_net, cec_indicator=cec_indicator)
            sim_net, removed_components = mrad_model.clean_network(sim_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            nonfunctional_component_size[i] = np.sum([len(removed_component) for removed_component in removed_components])
            sim_net = mrad_model.prepare_simulation_network(sim_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i] = get_lcc_size(sim_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                effective_conductances[i] = 0  
                functional_lcc_size[i] = 0
                nonfunctional_component_size[i] = len(net['pore.coords'])
            if (str(e) == "'throat.conns'") and (len(sim_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                effective_conductances[i] = 0
                functional_lcc_size[i] = 0
                nonfunctional_component_size[i] = len(net['pore.coords'])
    
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility

def get_lcc_size(net):
    """
    Calculates the size of the largest connected component of a network.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between them

    Returns
    -------
    lcc_size : int
        size of the largest connected component
    susceptibility : int
        size of the second-largest connected component
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
    component_sizes = np.sort(np.unique([len(np.where(component_labels == component_label)[0]) for component_label in np.unique(component_labels)]))
    lcc_size = component_sizes[-1]
    if len(component_sizes) > 1:
        susceptibility = component_sizes[-2]
    else:
        susceptibility = 0
    return lcc_size, susceptibility