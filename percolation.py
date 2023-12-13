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
import itertools

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
        si_type : str, defines how the decision on spreading (= removal of a conduit) is made: if 'stochastic', embolism spreads to 
        neighbouring conduits with a given probabability, while if 'physiological', embolism spreads if the air pressure
        at one side of a pit exceeds the threshold
        spreading_probability : double, probability at which embolism spreads to neighbouring conduits if si_type == 'stochastic'
        spreading_threshold : double, mimimum air pressure at one side of a pit required for spreading of the embolism if si_type == 'physiological'
        start_conduit : int or 'random', index of the first conduit to be removed (i.e. the first infected node of the simulation), if 'random', the start
        conduit is selected at random
    percolation type : str, optional
        type of the percolation, 'bond' to remove throats, 'site' to remove pores, 'conduit' to remove conduits, or
        'si' to simulate the spreading of embolism between conduits using the SI model(default: 'bond')
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
    nonfunctional_component_volume : np.array
        total volume of nonfunctional components
    prevalence : np.array
        the fraction of embolized conduits in the network (only calculated if percolation_type == 'si')
    """
    # TODO: calculate also the percentage of conductance lost to match Mrad's VC plots (although this is a pure scaling: current conductance divided by the original one)
    assert percolation_type in ['bond', 'site', 'conduit', 'si'], 'percolation type must be bond (removal of throats), site (removal of pores), conduit (removal of conduits), or si (removal of conduits as SI spreading process)'
    if percolation_type == 'conduit':
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_physiological_conduit_percolation(net, cfg)
    elif percolation_type == 'si':
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence = run_conduit_si(net, cfg, si_length=cfg['si_length'], start_conduit=cfg['start_conduit'], si_type=cfg['si_type'])
    elif break_nonfunctional_components:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_graph_theoretical_element_percolation(net, cfg, percolation_type, removal_order)
    else:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_physiological_element_percolation(net, cfg, percolation_type)
    if not percolation_type == 'si':
        prevalence = np.zeros(len(effective_conductances))
    
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence

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
    nonfunctional_component_volume : np.array
        the total volume of non-functional components
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
    nonfunctional_component_volume = np.zeros(n_removals)
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
            try:
                op.topotools.trim(sim_net, throats=removal_order[:i + 1])
            except Exception as e:
                if (str(e) == "'throat.conns'") and (len(net['throat.all']) == 0):
                    nonfunctional_component_size[i::] = len(net['pore.coords'])
                    break
        elif percolation_type == 'site':
            try:
                op.topotools.trim(sim_net, pores=removal_order[:i + 1])
            except Exception as e:
                if str(e) == 'Cannot delete ALL pores':
                    nonfunctional_component_size[i::] = len(net['pore.coords']) - i
                    break # this would remove the last node from the network; at this point, value of all outputs should be 0
        lcc_size[i], susceptibility[i] = get_lcc_size(sim_net)
        try:
            conduit_elements = mrad_model.get_conduit_elements(sim_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            pore_diameter = sim_net['pore.diameter']
            sim_net, removed_components = mrad_model.clean_network(sim_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            nonfunctional_component_size[i] = np.sum([len(removed_component) for removed_component in removed_components])
            removed_elements = list(itertools.chain.from_iterable(removed_components))
            nonfunctional_component_volume[i] = np.sum(np.pi * 0.5 * pore_diameter[removed_elements]**2 * conduit_element_length)
            n_inlets[i], n_outlets[i] = get_n_inlets(sim_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(sim_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i] = get_lcc_size(sim_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[i] = len(net['pore.coords'])
                nonfunctional_component_volume[i] = np.sum(np.pi * 0.5 * pore_diameter**2 * conduit_element_length)
            elif (str(e) == "'throat.conns'") and (len(sim_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i] = len(net['pore.coords'])
                nonfunctional_component_volume[i] = np.sum(np.pi * 0.5 * pore_diameter**2 * conduit_element_length)
            else:
                raise
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume

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
    # TODO: add a case for removing elements in given order instead of at random; however, this is not a high priority thing
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
    nonfunctional_component_volume = np.zeros(n_removals)
    cfg['conduit_diameters'] = 'inherit_from_net'
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']
    
    max_removed_lcc = 0
    
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
                    nonfunctional_component_size[i::] = len(net['pore.coords'])
                    lcc_size[i::] = max_removed_lcc
                    break
                    
        elif percolation_type == 'site':
            pore_to_remove = np.random.randint(perc_net['pore.coords'].shape[0] - 1)
            try:
                op.topotools.trim(perc_net, pores=pore_to_remove)
            except Exception as e:
                if str(e) == 'Cannot delete ALL pores':
                    nonfunctional_component_size[i::] = len(net['pore.coords']) - i
                    lcc_size[i::] = max_removed_lcc
                    break # this would remove the last node from the network; at this point, value of all outputs should be 0
        lcc_size[i], susceptibility[i] = get_lcc_size(perc_net)
        try:
            pore_diameter = perc_net['pore.diameter']
            conduit_elements = mrad_model.get_conduit_elements(perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            perc_net, removed_components = mrad_model.clean_network(perc_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            removed_elements = list(itertools.chain.from_iterable(removed_components))
            nonfunctional_component_volume[i] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * pore_diameter[removed_elements]**2 * conduit_element_length)
            if len(removed_components) > 0:
                removed_lcc = max([len(component) for component in removed_components])
                if removed_lcc > max_removed_lcc: # Percolation doesn't affect the sizes of removed components -> the largest removed component size changes only if a new, larger component gets removed
                    max_removed_lcc = removed_lcc
            if lcc_size[i] < max_removed_lcc:
                lcc_size[i] = max_removed_lcc
            nonfunctional_component_size[i] = nonfunctional_component_size[i - 1] + np.sum([len(removed_component) for removed_component in removed_components])
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i] = get_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                if percolation_type == 'site':
                    nonfunctional_component_size[i::] = len(net['pore.coords']) - i # all but removed elements are nonfunctional
                    nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * pore_diameter**2 * conduit_element_length)
                elif percolation_type == 'bond':
                    nonfunctional_component_size[i::] = len(net['pore.coords']) # all links have been removed; all elements are nonfunctional
                    nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * pore_diameter**2 * conduit_element_length)
                lcc_size[i::] = max_removed_lcc
                break
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                if percolation_type == 'site':
                    nonfunctional_component_size[i::] = len(net['pore.coords']) - i # all but removed elements are nonfunctional
                    nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * pore_diameter**2 * conduit_element_length)
                elif percolation_type == 'bond':
                    nonfunctional_component_size[i::] = len(net['pore.coords']) # all links have been removed; all elements are nonfunctional
                    nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * pore_diameter**2 * conduit_element_length)
                lcc_size[i::] = max_removed_lcc
                break
            else:
                raise
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume

def run_physiological_conduit_percolation(net, cfg, removal_order='random'):
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
    removal_order : iterable or str, optional (default='random')
        indices of pores to be removed in the order of removal. for performing full percolation analysis, len(removal_order)
        should match the number of pores. string 'random' can be given to indicate removal in random order.
        
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
    nonfunctional_component_volume : np.array
        the total volume of non-functional components
    """
    # TODO: What would bond percolation mean at the level of conduits?
        
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
        
    effective_conductances = np.zeros(n_removals)
    lcc_size = np.zeros(n_removals)
    functional_lcc_size = np.zeros(n_removals)
    nonfunctional_component_size = np.zeros(n_removals)
    nonfunctional_component_volume = np.zeros(n_removals)
    susceptibility = np.zeros(n_removals)
    functional_susceptibility = np.zeros(n_removals)
    n_inlets = np.zeros(n_removals)
    n_outlets = np.zeros(n_removals)
    cfg['conduit_diameters'] = 'inherit_from_net'
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']
        
    max_removed_lcc = 0
    
    for i in range(n_removals):
        try:
            if i > 0:
                conns = perc_net['throat.conns']
                conn_types = perc_net['throat.type']
                cec_mask = conn_types == cec_indicator
                cec = conns[cec_mask]
                conduits = mrad_model.get_conduits(cec)
            if removal_order == 'random':
                conduit_to_remove = conduits[np.random.randint(conduits.shape[0]), :]
            else:
                removal_order = np.array(removal_order)
                for index in np.arange(i, n_removals):
                    removal_index = removal_order[index]
                    if removal_index >= 0:
                        break
                conduit_to_remove = conduits[removal_index]
                removal_order[np.where(removal_order == removal_index)[0]] = -1
                removal_order[np.where(removal_order > removal_index)[0]] -= 1 
            pores_to_remove = np.arange(conduit_to_remove[0], conduit_to_remove[1] + 1)
            op.topotools.trim(perc_net, pores=pores_to_remove)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores':
                nonfunctional_component_size[i::] = n_removals - i
                nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1]
                lcc_size[i::] = max_removed_lcc
                break # this would remove the last node from the network; at this point, value of all outputs should be 0
            elif str(e) == "'throat.type'" and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i::] = n_removals
                nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1]
                lcc_size[i::] = max_removed_lcc
                break
            else:
                raise
        try:
            lcc_size[i], susceptibility[i], _ = get_conduit_lcc_size(perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns'])
            orig_perc_net['throat.type'] = perc_net['throat.type']
            orig_perc_net['pore.diameter'] = perc_net['pore.diameter']
            perc_net, removed_components = mrad_model.clean_network(perc_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            removed_elements = list(itertools.chain.from_iterable(removed_components)) # calculating the size of the largest removed component in conduits
            nonfunctional_component_volume[i] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'][removed_elements]**2 * conduit_element_length)
            if len(removed_elements) > 0:
                removed_net = get_induced_subnet(orig_perc_net, removed_elements)
                removed_lcc, _, n_nonfunctional_conduits = get_conduit_lcc_size(removed_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                       conduit_element_length=conduit_element_length, 
                                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
                nonfunctional_component_size[i] = nonfunctional_component_size[i - 1] + n_nonfunctional_conduits
                if removed_lcc > max_removed_lcc: # Percolation doesn't affect the sizes of removed components -> the largest removed component size changes only if a new, larger component gets removed
                    max_removed_lcc = removed_lcc
                if not removal_order == 'random':
                    removed_conduit_indices = np.unique(conduit_elements[removed_elements, 3])[::-1] - 1 # indexing of conduits in conduit_elements begins from 1
                    removal_order[np.where(np.array([removal_index in removed_conduit_indices for removal_index in removal_order]))[0]] = -1
                    removal_order = removal_order -  np.sum(np.array([removal_order > removed_conduit_index for removed_conduit_index in removed_conduit_indices]), axis=0)
            else:
                nonfunctional_component_size[i] = nonfunctional_component_size[i - 1]
            if max_removed_lcc > lcc_size[i]: # checking if the largest removed component is larger than the largest functional one
                lcc_size[i] = max_removed_lcc
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i], _ = get_conduit_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[i::] = n_removals - i
                nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter']**2 * conduit_element_length)
                lcc_size[i::] = max_removed_lcc
                break
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i::] = n_removals - i
                nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'] * conduit_element_length)
                lcc_size[i::] = max_removed_lcc
                break
            else:
                raise
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume

def run_conduit_si(net, cfg, start_conduit, si_length=1000, si_type='stochastic', spreading_probability=0.1, spreading_threshold=0):
    """
    Starting from a given conduit, simulates an SI (embolism) spreading process on the conduit network: at each step, each conduit is
    removed from the network at a certain possibility that depends on if their neighbours have been removed. The removal
    decision can be made either stochastically or based on simulated pressure at neighbouring conduits.

    Parameters
    ----------
    net : op.Network
        pores correspond to conduit elements, throats to connections between them
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
        spreading_probability : double, probability at which embolism spreads to neighbouring conduits if si_type == 'stochastic' (default: 0.1)
        spreading_threshold : double, mimimum air pressure at one side of a pit required for spreading of the embolism if si_type == 'physiological' (default: ???)
    start_conduit : int or 'random'
        index of the first conduit to be removed (i.e. the first infected node of the simulation), if 'random', the start
        conduit is selected at random
    si_length : int, optional
        number of time steps used for the simulation (default: 1000)
    si_type : str, optional
        defines how the decision on spreading (= removal of a conduit) is made: if 'stochastic', embolism spreads to 
        neighbouring conduits with a given probabability, while if 'physiological', embolism spreads if the air pressure
        at one side of a pit exceeds the threshold
    
    Returns
    -------
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
    nonfunctional_component_volume : np.array
        the total volume of non-functional components
    prevalence : np.array
        fraction of embolized conduits at each infection step

    """
    # TODO: pick a reasonable default value for si_length, spreading_probability, and spreading_threshold
    assert si_type in ['stochastic', 'physiological'], "Unknown si_type; select 'stochastic' or 'physiological'"
    # TODO: implement physiological SI where embolization spreads depending on air pressure in conduits
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    heartwood_d = cfg.get('heartwood_d', params.heartwood_d)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', False)
    spreading_probability = cfg.get('spreading_probability', 0.1)
    spreading_threshold = cfg.get('spreading_threshold', 100)
    
    conns = net['throat.conns']
    assert len(conns) > 0, 'Network has no throats; cannot run percolation analysis'
    conn_types = net['throat.type']
    cec_mask = conn_types == cec_indicator
    cec = conns[cec_mask]
    conduits = mrad_model.get_conduits(cec)
    orig_conduits = conduits.copy()
        
    effective_conductances = np.zeros(si_length)
    lcc_size = np.zeros(si_length)
    functional_lcc_size = np.zeros(si_length)
    nonfunctional_component_size = np.zeros(si_length)
    nonfunctional_component_volume = np.zeros(si_length)
    susceptibility = np.zeros(si_length)
    functional_susceptibility = np.zeros(si_length)
    n_inlets = np.zeros(si_length)
    n_outlets = np.zeros(si_length)
    prevalence = np.zeros(si_length)
    cfg['conduit_diameters'] = 'inherit_from_net'
    
    if start_conduit == 'random':
        start_conduit = np.random.randint(conduits.shape[0]) + 1 # indexing of conduits starts from 1, not from 0
    
    time = np.arange(si_length)
    embolization_times = np.zeros((conduits.shape[0], 2))
    embolization_times[:, 0] = np.inf
    embolization_times[start_conduit - 1, 0] = 0
    embolization_times[:, 1] = 1 # the second column indicates if the conduit is functional
    conduit_neighbours = get_conduit_neighbors(net, use_cylindrical_coords, conduit_element_length, heartwood_d, cec_indicator)
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']
        
    max_removed_lcc = 0
        
    for time_step in time:
            
        pores_to_remove = []
        removed_conduit_indices = []
        
        if time_step == 0:
            pores_to_remove.extend(list(np.arange(conduits[start_conduit - 1][0], conduits[start_conduit - 1][1] + 1)))
            conduits[start_conduit - 1, :] = -1
            conduits[start_conduit::, 0:2] = conduits[start_conduit::, 0:2] - len(pores_to_remove)
        else:
            for conduit in conduit_neighbours.keys():
                if embolization_times[conduit - 1, 0] < time_step:
                    continue # conduit is already embolized
                else:
                    if len(conduit_neighbours[conduit]) > 0:
                        neighbour_embolization_times = embolization_times[np.array(conduit_neighbours[conduit]) - 1, 0]
                    else:
                        neighbour_embolization_times = []
                    if np.any(neighbour_embolization_times < time_step): # there are embolized neighbours
                        if np.random.rand() > 1 - spreading_probability:
                            embolization_times[conduit - 1, 0] = time_step
                            if embolization_times[conduit - 1, 1] > 0: # if conduit is functional, it will be removed from the simulation network
                                embolization_times[conduit - 1, 1] = 0
                                conduit_pores = np.arange(conduits[conduit - 1, 0], conduits[conduit - 1, 1] + 1)
                                removed_conduit_indices.append(conduit - 1)
                                pores_to_remove.extend(list(conduit_pores))
                            else: # if a nonfunctional conduit is embolized, nonfunctional component size and volume decrease
                                nonfunctional_component_size[time_step] -= 1
                                nonfunctional_component_volume[time_step] -= np.sum(np.pi * 0.5 * net['pore.diameter'][np.arange(orig_conduits[conduit - 1, 0], orig_conduits[conduit - 1, 1] + 1)]**2 * conduit_element_length)
            for removed_conduit_index in np.sort(removed_conduit_indices)[::-1]:
                conduits[removed_conduit_index + 1::, 0:2] = conduits[removed_conduit_index + 1::, 0:2] - conduits[removed_conduit_index, 2]
            conduits[removed_conduit_indices, :] = -1
        op.topotools.trim(perc_net, pores=pores_to_remove)
        prevalence[time_step] = np.sum(embolization_times[:, 0] <= time_step) / conduits.shape[0]

        try:
            lcc_size[time_step], susceptibility[time_step], _ = get_conduit_lcc_size(perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns'])
            orig_perc_net['throat.type'] = perc_net['throat.type']
            orig_perc_net['pore.diameter'] = perc_net['pore.diameter']
            perc_net, removed_components = mrad_model.clean_network(perc_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            removed_elements = list(itertools.chain.from_iterable(removed_components)) # calculating the size of the largest removed component in conduits
            nonfunctional_component_volume[time_step] += nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'][removed_elements]**2 * conduit_element_length)
            if len(removed_elements) > 0:
                removed_net = get_induced_subnet(orig_perc_net, removed_elements)
                removed_lcc, _, n_nonfunctional_conduits = get_conduit_lcc_size(removed_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                       conduit_element_length=conduit_element_length, 
                                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
                nonfunctional_component_size[time_step] = nonfunctional_component_size[time_step] + nonfunctional_component_size[time_step - 1] + n_nonfunctional_conduits
                if removed_lcc > max_removed_lcc: # Percolation doesn't affect the sizes of removed components -> the largest removed component size changes only if a new, larger component gets removed
                    max_removed_lcc = removed_lcc
                removed_conduit_indices = get_conduit_indices(conduits, removed_elements)
                for removed_conduit_index in removed_conduit_indices[::-1]:
                    conduits[removed_conduit_index + 1::, 0:2] = conduits[removed_conduit_index + 1::, 0:2] - conduits[removed_conduit_index, 2]
                    conduits[removed_conduit_index, :] = -1
                embolization_times[removed_conduit_indices, 1] = 0
            else:
                nonfunctional_component_size[time_step] = nonfunctional_component_size[time_step] + nonfunctional_component_size[time_step - 1]
            if max_removed_lcc > lcc_size[time_step]: # checking if the largest removed component is larger than the largest functional one
                lcc_size[time_step] = max_removed_lcc
            n_inlets[time_step], n_outlets[time_step] = get_n_inlets(perc_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[time_step] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[time_step], functional_susceptibility[time_step], _ = get_conduit_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[time_step::] = net['pore.coords'].shape[0] - time_step
                nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter']**2 * conduit_element_length)
                lcc_size[time_step::] = max_removed_lcc
                break
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[time_step::] = net['pore.coords'].shape[0] - time_step
                nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'] * conduit_element_length)
                lcc_size[time_step::] = max_removed_lcc
                break
            else:
                raise
    cut_index = np.where(prevalence == prevalence[-1])[0][0] + 1
    return effective_conductances[0:cut_index], lcc_size[0:cut_index], functional_lcc_size[0:cut_index], nonfunctional_component_size[0:cut_index], susceptibility[0:cut_index], functional_susceptibility[0:cut_index], n_inlets[0:cut_index], n_outlets[0:cut_index], nonfunctional_component_volume[0:cut_index], prevalence[0:cut_index]
    

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
    n_conduits : int
        number conduits of the network 
    """
    conduit_elements = mrad_model.get_conduit_elements(net, use_cylindrical_coords=use_cylindrical_coords, 
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    throat_conduits = []
    conduit_indices = []
    throats = net.get('throat.conns', [])
    if len(throats) > 0:
        for i, throat in enumerate(throats):
            start_conduit = conduit_elements[throat[0], 3] # NOTE: indexing of conduits start from 1 instead of 0
            end_conduit = conduit_elements[throat[1], 3]
            if not start_conduit in conduit_indices:
                conduit_indices.append(start_conduit)
            if not end_conduit in conduit_indices:
                conduit_indices.append(end_conduit)
            if not start_conduit == end_conduit:
                throat_conduits.append((start_conduit, end_conduit))
        n_conduits = len(conduit_indices)
        G = nx.Graph()
        G.add_nodes_from(conduit_indices)
        G.add_edges_from(throat_conduits)
        component_sizes = [len(component) for component in sorted(nx.connected_components(G), key=len, reverse=True)]
        lcc_size = component_sizes[0]
        if len(component_sizes) > 1:
            susceptibility = component_sizes[1]
        else:
            susceptibility = 0
    else: # each conduit element forms a conduit of its own
        lcc_size = 1
        susceptibility = 0
        n_conduits = net['pore.coords'].shape[0]
    return lcc_size, susceptibility, n_conduits

def get_conduit_neighbors(net, use_cylindrical_coords=False, conduit_element_length=params.Lce, 
                         heartwood_d=params.heartwood_d, cec_indicator=params.cec_indicator):
    """
    Finds the neighbors of each conduit (set of adjacent conduit elements in row/z direnction)

    Parameters
    ----------
    net : op.Network()
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
    conduit_neighbours : dict
        for each conduits, indices of its neighbours
    """
    conduit_elements = mrad_model.get_conduit_elements(net, use_cylindrical_coords=use_cylindrical_coords, 
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    
    conduit_neighbours = {int(i):[] for i in np.unique(conduit_elements[:, 3])}
    
    for i, throat in enumerate(net['throat.conns']):
        start_conduit = int(conduit_elements[throat[0], 3]) # NOTE: indexing of conduits start from 1 instead of 0
        end_conduit = int(conduit_elements[throat[1], 3])
        if not start_conduit == end_conduit:
            if not start_conduit in conduit_neighbours[end_conduit]:
                conduit_neighbours[end_conduit].append(start_conduit)
            if not end_conduit in conduit_neighbours[start_conduit]:
                conduit_neighbours[start_conduit].append(end_conduit)
    return conduit_neighbours
    
def get_induced_subnet(net, elements):
    """
    Constructs the subgraph of an op.Network spanned by a given set of pores (conduit elements).

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements and throats to connections between them
    elements : list of ints
        the conduit elements (pores) that span the subnetwork

    Returns
    -------
    subnet : op.Network()
        the subnetwork induced by elements
    """
    coords = net['pore.coords']
    conns = net['throat.conns']
    conn_types = net['throat.type']
    elements = np.sort(elements)
    subcoords = coords[elements, :]
    subconns = []
    subtypes = []
    for conn, conn_type in zip(conns, conn_types):
        if (conn[0] in elements) and (conn[1] in elements):
            subconns.append(np.array([np.where(elements == conn[0])[0][0], np.where(elements == conn[1])[0][0]]))
            subtypes.append(conn_type)
    subconns = np.array(subconns)
    subtypes = np.array(subtypes)
    if subconns.shape[0] > 0:
        subnet = op.network.Network(conns=subconns, coords=subcoords)
        subnet['throat.type'] = subtypes
    else:
        subnet = op.network.Network(coords=subcoords)
    return subnet

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
        
def get_conduit_indices(conduits, elements):
    """
    Finds the indices of the conduits (sets of consecutive elements in the row/z direction) to which given elements belong to.

    Parameters
    ----------
    conduits : np.array
        three columns: 1) the first element of the conduit, 2) the second element of the conduit, 3) size of the conduit
    elements : iterable of ints
        list of elements; each element should belong to one of the conduits 

    Returns
    -------
    conduit_indices : np.array
        for each element, the index of the conduit to which the element belongs to
    """
    conduit_indices = []
    for element in elements:
        conduit_index = np.where((conduits[:, 0] <= element) & (conduits[:, 1] >= element))[0][0]
        conduit_indices.append(conduit_index)
    conduit_indices = np.unique(conduit_indices)
    return conduit_indices
            
    
            
    