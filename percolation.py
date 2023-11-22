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

import mrad_model
import mrad_params as params

def run_percolation(net, cfg, percolation_type='bond', removal_order='random', break_nonfunctional_components=True):
    """
    Removes throats (bond percolation) or pores (site percolation) from an OpenPNM network object in a given (or random) order and calculates 
    a number of measures, including effective conductance and largest component size, after removal (see below for details). If order is given,
    percolation is "graph-theoretical", i.e. nonfunctional components (i.e. components not connected to inlet or outlet) can
    be further broken. Percolation with random order can be also "physiological", i.e. nonfunctional components are not
    broken further.

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
        conduit_element_length : float, length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
        heartwood_d : float, diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements) used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
    percolation type : str, optional
        type of the percolation, 'bond' to remove throats, 'site' to remove pores, or 'conduit' to remove conduits (default: 'bond')
    removal_order : iterable or str, optional (default='random')
        indices of pores to be removed in the order of removal. for performing full percolation analysis, len(removal_order)
        should match the number of pores. string 'random' can be given to indicate removal in random order.
    break_nonfunctional_components : bln, optional
        can links/nodes in components that are nonfunctional (i.e. not connected to inlet or outlet) be removed (default: True)

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
    n_inlets : np.array
        the mean number of inlet elements per functional component
    n_outlets : np.array
        the mean number of outlet elements per functional component
    """
    # TODO: calculate also the percentage of conductance lost to match Mrad's VC plots (although this is a pure scaling: current conductance divided by the original one)
    assert percolation_type in ['bond', 'site', 'conduit'], 'percolation type must be bond (removal of throats), site (removal of pores), or conduit (removal of conduits'
    if percolation_type == 'conduit':
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets = run_physiological_conduit_percolation(net, cfg)
    elif break_nonfunctional_components:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets = run_graph_theoretical_element_percolation(net, cfg, percolation_type, removal_order)
    else:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets = run_physiological_element_percolation(net, cfg, percolation_type)
    
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets

def run_graph_theoretical_element_percolation(net, cfg, percolation_type='bond', removal_order='random'):
    """
    Removes links (bond percolation) or nodes (site percolation) from an openpnm network object in a given (or random)
    order. 

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
        conduit_element_length : float, length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
        heartwood_d : float, diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements) used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
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
    n_inlets : np.array
        the mean number of inlet elements per functional component
    n_outlets : np.array
        the mean number of outlet elements per functional component
    """
    if percolation_type == 'bond':
        n_removals = net['throat.conns'].shape[0]
    elif percolation_type == 'site':
        n_removals = net['pore.coords'].shape[0]
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    heartwood_d = cfg.get('heartwood_d', params.heartwood_d)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', False)
    
    effective_conductances = np.zeros(n_removals)
    lcc_size = np.zeros(n_removals)
    functional_lcc_size = np.zeros(n_removals)
    nonfunctional_component_size = np.zeros(n_removals)
    susceptibility = np.zeros(n_removals)
    functional_susceptibility = np.zeros(n_removals)
    n_inlets = np.zeros(n_removals)
    n_outlets = np.zeros(n_removals)
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
            conduit_elements = mrad_model.get_conduit_elements(sim_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            sim_net, removed_components = mrad_model.clean_network(sim_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            nonfunctional_component_size[i] = np.sum([len(removed_component) for removed_component in removed_components])
            n_inlets[i], n_outlets[i] = get_n_inlets(sim_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(sim_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i] = get_lcc_size(sim_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[i] = len(net['pore.coords'])
            elif (str(e) == "'throat.conns'") and (len(sim_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i] = len(net['pore.coords'])
            else:
                raise
    
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets

def run_physiological_element_percolation(net, cfg, percolation_type):
    """
    Removes links (bond percolation) or nodes (site percolation) of an openpnm network object in random order till there
    are no links (nodes) left. Components are removed from the network as soon as they become nonfunctional 
    (i.e. are not connected to inlet or outlet) are removed
    

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
        conduit_element_length : float, length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
        heartwood_d : float, diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements) used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
    percolation type : str, optional
        type of the percolation, 'bond' to remove throats or 'site' to remove pores (default: 'bond')

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
    n_inlets : np.array
        the mean number of inlet elements per functional component
    n_outlets : np.array
        the mean number of outlet elements per functional component
    """
    # TODO: add a case for removing pores/throats in a given order instead of the random one
    if percolation_type == 'bond':
        n_removals = net['throat.conns'].shape[0]
    elif percolation_type == 'site':
        n_removals = net['pore.coords'].shape[0]
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    heartwood_d = cfg.get('heartwood_d', params.heartwood_d)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', False)
    
    effective_conductances = np.zeros(n_removals)
    lcc_size = np.zeros(n_removals)
    functional_lcc_size = np.zeros(n_removals)
    nonfunctional_component_size = np.zeros(n_removals)
    susceptibility = np.zeros(n_removals)
    functional_susceptibility = np.zeros(n_removals)
    n_inlets = np.zeros(n_removals)
    n_outlets = np.zeros(n_removals)
    cfg['conduit_diameters'] = 'inherit_from_net'
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']
    
    for i in range(n_removals):
        if percolation_type == 'bond':
            try:
                if len(perc_net['throat.conns']) > 1:
                    throat_to_remove = np.random.randint(perc_net['throat.conns'].shape[0] - 1)
                    op.topotools.trim(perc_net, throats=throat_to_remove)
                else:
                    op.topotools.trim(perc_net, throats=0)
            except KeyError as e:
                if str(e) == "'throat.conns'" and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                    nonfunctional_component_size[i] = len(net['pore.coords'])
                    
        elif percolation_type == 'site':
            pore_to_remove = np.random.randint(perc_net['pore.coords'].shape[0] - 1)
            try:
                op.topotools.trim(perc_net, pores=pore_to_remove)
            except Exception as e:
                if str(e) == 'Cannot delete ALL pores':
                    nonfunctional_component_size[i] = len(net['pore.coords'])
                    continue # this would remove the last node from the network; at this point, value of all outputs should be 0
        lcc_size[i], susceptibility[i] = get_lcc_size(perc_net)
        try:
            conduit_elements = mrad_model.get_conduit_elements(perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            perc_net, removed_components = mrad_model.clean_network(perc_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            nonfunctional_component_size[i] = nonfunctional_component_size[i - 1] + np.sum([len(removed_component) for removed_component in removed_components])
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i] = get_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[i] = len(net['pore.coords'])
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i] = len(net['pore.coords'])
            else:
                raise
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets

def run_physiological_conduit_percolation(net, cfg):
    """
    Removes conduits (sets of consequtive conduit elements in row/z direction) with all their ICC throats from an OpenPNM
    network object till there are no conduits left. Components are removed from the network as soon as they become nonfunctional 
    (i.e. are not connected to inlet or outlet) are removed.
    
    Parameters:
    -----------
    net : op.Network
        pores correspond to conduit elements and throats to connections between them
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
        conduit_element_length : float, length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
        heartwood_d : float, diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements) used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
    Returns:
    --------
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
    n_inlets : np.array
        the mean number of inlet elements per functional component
    n_outlets : np.array
        the mean number of outlet elements per functional component
    """
    # TODO: add case for removing conduits in a given order instead of a random order
    # What would bond percolation mean at the level of conduits?
    # A spreading model could be a better model for the phenomenon than random percolation
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    heartwood_d = cfg.get('heartwood_d', params.heartwood_d)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', False)
    
    conns = net['throat.conns']
    assert len(conns) > 0, 'Network has no throats; cannot run percolation analysis'
    conn_types = net['throat.type']
    cec_mask = conn_types == cec_indicator
    cec = conns[cec_mask]
    conduits = mrad_model.get_conduits(cec)
    n_removals = conduits.shape[0]
    n_previous_conduits = conduits.shape[0]
        
    effective_conductances = np.zeros(n_removals)
    lcc_size = np.zeros(n_removals)
    functional_lcc_size = np.zeros(n_removals)
    nonfunctional_component_size = np.zeros(n_removals)
    susceptibility = np.zeros(n_removals)
    functional_susceptibility = np.zeros(n_removals)
    n_inlets = np.zeros(n_removals)
    n_outlets = np.zeros(n_removals)
    cfg['conduit_diameters'] = 'inherit_from_net'
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']
    
    for i in range(n_removals):
        try:
            if i > 0:
                conns = perc_net['throat.conns']
                conn_types = perc_net['throat.type']
                cec_mask = conn_types == cec_indicator
                cec = conns[cec_mask]
                conduits = mrad_model.get_conduits(cec)
                nonfunctional_component_size[i - 1] = nonfunctional_component_size[i - 2] + n_previous_conduits - conduits.shape[0]
            n_previous_conduits = conduits.shape[0] - 1
            conduit_to_remove = conduits[np.random.randint(conduits.shape[0]), :]
            pores_to_remove = np.arange(conduit_to_remove[0], conduit_to_remove[1] + 1)
            op.topotools.trim(perc_net, pores=pores_to_remove)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores':
                nonfunctional_component_size[i - 1] = n_removals
                continue # this would remove the last node from the network; at this point, value of all outputs should be 0
            elif str(e) == "'throat.type'" and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i - 1] = n_removals
                continue
        try:
            lcc_size[i], susceptibility[i] = get_conduit_lcc_size(perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            perc_net, removed_components = mrad_model.clean_network(perc_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i] = get_conduit_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[i - 1] = n_removals
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i - 1] = n_removals
            else:
                raise
    nonfunctional_component_size[-1] = n_removals # when all conduits have been removed, there are no functional components
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets

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
    _, _, component_sizes = mrad_model.get_components(net)
    component_sizes = np.sort(component_sizes)
    lcc_size = component_sizes[-1]
    if len(component_sizes) > 1:
        susceptibility = component_sizes[-2]
    else:
        susceptibility = 0
    return lcc_size, susceptibility

def get_conduit_lcc_size(net, use_cylindrical_coords=False, conduit_element_length=params.Lce, 
                         heartwood_d=params.heartwood_d, cec_indicator=params.cec_indicator):
    """
    Calculates the largest connected component size and susceptibility in a network where nodes correspond to conduits
    (= sets of conduit elements connected in row/z direction).

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements and throats to connections between them
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
    lcc_size : int
        size of the largest connected component
    susceptibility : int
        size of the second-largest connected component
    """
    conduit_elements = mrad_model.get_conduit_elements(net, use_cylindrical_coords=use_cylindrical_coords, 
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    throat_conduits = []
    conduit_indices = []
    for i, throat in enumerate(net['throat.conns']):
        start_conduit = conduit_elements[throat[0], 3]
        end_conduit = conduit_elements[throat[1], 3]
        if not start_conduit in conduit_indices:
            conduit_indices.append(start_conduit)
        if not end_conduit in conduit_indices:
            conduit_indices.append(end_conduit)
        if not start_conduit == end_conduit:
            throat_conduits.append((start_conduit, end_conduit))
    G = nx.Graph()
    G.add_nodes_from(conduit_indices)
    G.add_edges_from(throat_conduits)
    component_sizes = [len(component) for component in sorted(nx.connected_components(G), key=len, reverse=True)]
    lcc_size = component_sizes[0]
    if len(component_sizes) > 1:
        susceptibility = component_sizes[1]
    else:
        susceptibility = 0
    return lcc_size, susceptibility

def get_n_inlets(net, outlet_row_index, cec_indicator=params.cec_indicator, conduit_element_length=params.Lce, 
                 heartwood_d=params.heartwood_d, use_cylindrical_coords=False):
    """
    Calculates the mean number of inlet and outlet elements per functional component (i.e. component with at least one
    inlet and one outlet element).
    
    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between them
    outlet_row_index : int
        index of the last row of the network (= n_rows - 1)
    cec_indicator : int, optional 
        value used to indicate that the type of a throat is CEC (default value from the Mrad Matlab implementation)
    conduit_element_length : float, optional
        length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
    heartwood_d : float, optional
        diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements)
        used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
    use_cylindrical_coords : bln, optional
        should Mrad model coordinates be interpreted as cylindrical ones

    Returns
    -------
    n_inlets : float
        average number of inlets per functional component
    n_outlets : float
        average number of outlets per functional component
    """
    _, component_indices, _ = mrad_model.get_components(net)
    conduit_elements = mrad_model.get_conduit_elements(net, use_cylindrical_coords=use_cylindrical_coords,
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    n_inlets = 0
    n_outlets = 0
    n_functional = 0
    for component in component_indices:
        in_btm = np.sum(conduit_elements[component, 0] == 0)
        in_top = np.sum(conduit_elements[component, 0] == outlet_row_index)
        if (in_btm > 0) & (in_top > 0):
            n_inlets += in_btm
            n_outlets += in_top
            n_functional += 1
    n_inlets = n_inlets / n_functional
    n_outlets = n_outlets / n_functional
    return n_inlets, n_outlets
        
            
    
            
    