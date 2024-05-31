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
import functools
from concurrent.futures import ProcessPoolExecutor as Pool

import mrad_model
import mrad_params as params
import simulations

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
        conduit_diameters: np.array of floats, diameters of the conduits, 'inherit_from_net' to use the conduit diameters from net, or 'lognormal' to draw diameters from a lognormal 
        distribution defined by Dc and Dc_cv. NOTE: if diameters have been defined in net (i.e. net['pore.diameter'] != []), conduit_diameters is omitted and the diameters from net are used.
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
        spreading_probability : double, probability at which embolism spreads to neighbouring conduits
        start_conduits : array-like of ints, 'random', or 'bottom', index of the first conduit to be removed (i.e. the first infected node of the simulation), if 'random', the start
        conduit is selected at random, if 'bottom', all conduits with an inlet pore are used
        pressure : int or array-like, the frequency steps to investigate (if an int is given, a log-range with the corresponding number of steps is used)
        si_type : str, 'stochastic' for probability-based spreading, 'physiological' for spreading based on pressure differences (default stochastic)
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading)
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading)
    percolation type : str, optional
        type of the percolation, 'bond' to remove throats, 'site' to remove pores, 'conduit' to remove conduits,
        'si' to simulate the spreading of embolism between conduits using the SI model or 'drainage' to simulate the spreading of embolism with the openpnm drainage algorithm (default: 'bond')
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
    if 'pore.diameter' in net.keys() and len(net['pore.diameter']) > 0:
        cfg['conduit_diameters'] == 'inherit_from_net'
    assert percolation_type in ['bond', 'site', 'conduit', 'si', 'drainage'], 'percolation type must be bond (removal of throats), site (removal of pores), conduit (removal of conduits), si (removal of conduits as SI spreading process), or drainage (removal of conduits using the openpnm drainage algorithm)'
    if percolation_type == 'conduit':
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_physiological_conduit_percolation(net, cfg)
    elif percolation_type == 'si':
        if cfg['si_type'] == 'stochastic':
            spreading_param = cfg.get('spreading_probability', 0.1)
        elif cfg['si_type'] == 'physiological':
            spreading_param = cfg.get('pressure_diff', 0)
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence = run_conduit_si(net, cfg, spreading_param)
    elif percolation_type == 'drainage':
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence = run_physiological_conduit_drainage(net, cfg, start_conduits=cfg['start_conduits'])
    elif break_nonfunctional_components:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_graph_theoretical_element_percolation(net, cfg, percolation_type, removal_order)
    else:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_physiological_element_percolation(net, cfg, percolation_type)
    if not percolation_type in ['si', 'drainage']:
        prevalence = np.zeros(len(effective_conductances))
    
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence

def construct_vulnerability_curve(net, cfg, x_range, start_conduits, si_length=1000):
    """
    Constructs the vulnerability curve (VC) of a network over a parameter range. At each parameter value, SI embolization spreading simulation is performed and effective
    conductance at the end of the spreading is calculated. The VC shows the percentage of effective conductance lost compared to the maximum value. 
    
    Definition of the x_range parameter depend on the SI type: in stochastic SI, the parameter is the spreading probability between a pair of conduits, while in physiological
    SI the parameter is the pressure difference between bubble (air) and water pressures.
    

    Parameters
    ----------
    net : openpnm.Network()
        a network object, pores correspond to conduit elements and throats to CECs and ICCs between them.
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
        si_tolerance_length : int, tolerance parameter for defining the end of the simulation: when the prevalence hasn't changed for si_tolerance_length steps, the simulation stops (default 20)
        si_type : str, 'stochastic' for probability-based spreading, 'physiological' for spreading based on pressure differences (default stochastic)
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6)
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2)
        average_pit_area : float, the average area of a pit
        nCPUs : int, number of CPUs used for parallel computing (default 5)
    x_range : np.array
        x axis values of the VC; spreading probabilities in case of stochastic SI and pressure differences in case of physiological SI
    start_conduits : str or array-like of ints
        the first conduits to be removed (i.e. the first infected node of the simulation)
        if 'random', a single start conduit is selected at random
        if 'random_per_component', a single start conduit per network component is selected at random
        if an array-like of ints is given, the ints are used as indices of the start conduits
    si_length : int, optional
        maximum number of time steps used for the simulation (default: 1000)

    Returns
    -------
    vc : tuple or np.arrays
        (x_range, vulnerability value at each x_range value)
    """    
    if x_range[0] > 0:
        x_range = np.concatenate((np.array([0]), x_range))
    nCPUs = cfg.get('nCPUs', 5)
    cfg['si_length'] = si_length
    
    if start_conduits in ['random', 'random_per_component']:
        conns = net['throat.conns']
        assert len(conns) > 0, 'Network has no throats; cannot run percolation analysis'
        conn_types = net['throat.type']
        cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
        cec_mask = conn_types == cec_indicator
        cec = conns[cec_mask]
        conduits = mrad_model.get_conduits(cec)
    
        if start_conduits == 'random':
            start_conduit = np.random.randint(conduits.shape[0])
            start_conduits = np.array([start_conduit])
        else: # start_conduits == 'random_per_component'
            _, component_indices, _ = mrad_model.get_components(net)
            start_conduits = np.zeros(len(component_indices), dtype=int)
            for i, component_elements in enumerate(component_indices):
                start_element = np.random.choice(component_elements)
                start_conduits[i] = np.where((conduits[:, 0] <= start_element) & (start_element <= conduits[:, 1]))[0][0]
    cfg['start_conduits'] = start_conduits
            
    vulnerability = np.zeros(x_range.shape)
    pool = Pool(max_workers = nCPUs)
    output = list(pool.map(run_conduit_si, itertools.repeat(net), itertools.repeat(cfg), x_range)) # simulating embolization spreading with each x_range value in parallel
    vulnerability = np.array([output_per_param[0][-1] for output_per_param in output]) # reading the last effective conductance value of each spreading simulation 
    vulnerability = 100 * (1 - vulnerability / vulnerability[0]) # normalization to obtain the percentage of conductance lost compared to the first x_range value
    vc = (x_range, vulnerability)
    return vc

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
        conduit_diameters: np.array of floats, diameters of the conduits, 'inherit_from_net' to use the conduit diameters from net, or 'lognormal' to draw diameters from a lognormal 
        distribution defined by Dc and Dc_cv. NOTE: if diameters have been defined in net (i.e. net['pore.diameter'] != []), conduit_diameters is omitted and the diameters from net are used.
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
    assert len(net['pore.diameter']) > 0, 'pore diameters not defined; please define pore diameters before running percolation'
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
            effective_conductances[i], _ = simulations.simulate_water_flow(sim_net, cfg, visualize=False)
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
    assert len(net['pore.diameter']) > 0, 'pore diameters not defined; please define pore diameters before running percolation'
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
            effective_conductances[i], _ = simulations.simulate_water_flow(sim_net, cfg, visualize=False)
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
    (i.e. are not connected to inlet or outlet).
    
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
       
    assert len(net['pore.diameter']) > 0, 'pore diameters not defined; please define pore diameters before running percolation'
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
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns']) # orig_perc_net is needed for calculating measures related to non-functional components (these are removed from perc_net before the simulations)
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
                    removed_conduit_indices = np.unique(conduit_elements[removed_elements, 3])[::-1]
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
            effective_conductances[i], _ = simulations.simulate_water_flow(sim_net, cfg, visualize=False)
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

def run_conduit_si(net, cfg, spreading_param=0):
    """
    Starting from a given conduit, simulates an SI (embolism) spreading process on the conduit network. The spreading can be stochastic (at each step, each conduit is
    embolized at a certain probability that depends on if their neighbours have been removed) or physiological (at each step, each conduit is
    embolized if it has embolized neighbours and the pressure difference with the neighbours is large enough) 

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
        si_tolerance_length : int, tolerance parameter for defining the end of the simulation: when the prevalence hasn't changed for si_tolerance_length steps, the simulation stops (default 20)
        si_type : str, 'stochastic' for probability-based spreading, 'physiological' for spreading based on pressure differences (default stochastic)
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6)
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2)
        average_pit_area : float, the average area of a pit
        start_conduits : str or array-like of ints, the first conduits to be removed (i.e. the first infected node of the simulation)
            if 'random', a single start conduit is selected at random
            if 'random_per_component', a single start conduit per network component is selected at random
            if an array-like of ints is given, the ints are used as indices of the start conduits
        si_length : int, maximum number of time steps used for the simulation (default: 1000)
    spreading_param : float
        parameter that controls the spreading speed, specifically
        if si_type == 'stochastic', spreading_param is the probability at which embolism spreads to neighbouring conduits (default: 0.1)
        if si_type == 'physiological', spreading param is difference between water pressure and vapour-air bubble pressure, delta P in the Mrad et al. article (default 0)
    
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
    # TODO: pick a reasonable default value for si_length and spreading_probability
    assert len(net['pore.diameter']) > 0, 'pore diameters not defined; please define pore diameters before running percolation'
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    heartwood_d = cfg.get('heartwood_d', params.heartwood_d)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', False)
    si_tolerance_length = cfg.get('si_tolerance_length', 20)
    si_type = cfg.get('si_type', 'stochastic')
    si_length = cfg.get('si_length', 1000)
    
    assert si_type in ['stochastic', 'physiological'], 'unknown si type, select stochastic or physiological'
    if si_type == 'stochastic':
        if spreading_param > 0:
            spreading_probability = spreading_param
        else:
            spreading_probability = 0.1
    elif si_type == 'physiological':
        pressure_diff = spreading_param
    
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
    
    start_conduits = cfg['start_conduits']
    if isinstance(start_conduits, str):
        if start_conduits == 'random':
            start_conduit = np.random.randint(conduits.shape[0])
            start_conduits = np.array([start_conduit])
        elif start_conduits == 'random_per_component':
            _, component_indices, _ = mrad_model.get_components(net)
            start_conduits = np.zeros(len(component_indices), dtype=int)
            for i, component_elements in enumerate(component_indices):
                start_element = np.random.choice(component_elements)
                start_conduits[i] = np.where((conduits[:, 0] <= start_element) & (start_element <= conduits[:, 1]))[0][0]
    
    embolization_times = np.zeros((conduits.shape[0], 2))
    embolization_times[:, 0] = np.inf
    embolization_times[:, 1] = 1 # the second column indicates if the conduit is functional
    for start_conduit in start_conduits:
        embolization_times[start_conduit, 0] = 0
        embolization_times[start_conduit, 1] = 0
    conduit_neighbours = get_conduit_neighbors(net, use_cylindrical_coords, conduit_element_length, heartwood_d, cec_indicator)
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']
    if si_type == 'physiological':
        bpp = calculate_bpp(perc_net, conduits, 1 - cec_mask, cfg)
        conduit_neighbour_bpp = {}
        for i, conduit in enumerate(conduits):
            iccs = conns[np.where(1 - cec_mask)]
            conduit_iccs = np.where(((conduit[0] <= iccs[:, 0]) & (iccs[:, 0] <= conduit[1])) | ((conduit[0] <= iccs[:, 1]) & (iccs[:, 1] <= conduit[1])))[0]
            neighbours = conduit_neighbours[i]
            neighbour_bpps = {}
            for neighbour in neighbours:
                neighbour_iccs = np.where(((conduits[neighbour, 0] <= iccs[:, 0]) & (iccs[:, 0] <= conduits[neighbour, 1])) | ((conduits[neighbour, 0] <= iccs[:, 1]) & (iccs[:, 1] <= conduits[neighbour, 1])))[0]
                shared_iccs = np.intersect1d(conduit_iccs, neighbour_iccs)
                shared_bpp = bpp[shared_iccs]
                neighbour_bpps[neighbour] = np.amin(shared_bpp)
            conduit_neighbour_bpp[i] = neighbour_bpps
        
    max_removed_lcc = 0
    time_step = 0
    prevalence_diff = 1
       
    while prevalence_diff > 0:
            
        pores_to_remove = []
        removed_conduit_indices = []
        
        if time_step == 0:
            for start_conduit in np.sort(start_conduits)[::-1]:
                pores_to_remove.extend(list(np.arange(conduits[start_conduit][0], conduits[start_conduit][1] + 1)))
                conduits[start_conduit +1::, 0:2] = conduits[start_conduit +1::, 0:2] - conduits[start_conduit, 2]
                conduits[start_conduit, :] = -1
        else:
            embolized_conduits = np.where(embolization_times[:, 0] < time_step)[0]
            possible_embolizations = False
            for embolized_conduit in embolized_conduits:
                try: 
                    neighbour_embolization_times = embolization_times[np.array(conduit_neighbours[embolized_conduit]), 0]
                except:
                    if len(conduit_neighbours[embolized_conduit]) == 0:
                        continue
                    else:
                        raise 
                if np.any(neighbour_embolization_times > time_step):
                    possible_embolizations = True
                    break
            if not possible_embolizations:
                break # no further embolizations are possible and the simulation stops
            
            for conduit in conduit_neighbours.keys():
                if embolization_times[conduit, 0] < time_step:
                    continue # conduit is already embolized
                else:
                    if len(conduit_neighbours[conduit]) > 0:
                        neighbour_embolization_times = embolization_times[np.array(conduit_neighbours[conduit]), 0]
                    else:
                        neighbour_embolization_times = np.array([])
                    if np.any(neighbour_embolization_times < time_step): # there are embolized neighbours
                        neighbours = conduit_neighbours[conduit]
                        embolized_neighbours = np.intersect1d(embolized_conduits, neighbours)
                        embolize = False
                        if si_type == 'stochastic':
                            embolize =  (np.random.rand() > (1 - spreading_probability)**(len(embolized_neighbours)))
                            #embolize = (np.random.rand() > 1 - spreading_probability)
                        elif si_type == 'physiological':
                            neighbour_bpp = np.array([conduit_neighbour_bpp[conduit][neighbour] for neighbour in embolized_neighbours])
                            embolize =  np.any(neighbour_bpp <= pressure_diff)
                        if embolize:
                            embolization_times[conduit, 0] = time_step
                            if embolization_times[conduit, 1] > 0: # if conduit is functional, it will be removed from the simulation network
                                embolization_times[conduit, 1] = 0
                                conduit_pores = np.arange(conduits[conduit, 0], conduits[conduit, 1] + 1)
                                removed_conduit_indices.append(conduit)
                                pores_to_remove.extend(list(conduit_pores))
                            else: # if a nonfunctional conduit is embolized, nonfunctional component size and volume decrease
                                nonfunctional_component_size[time_step] -= 1
                                nonfunctional_component_volume[time_step] -= np.sum(np.pi * 0.5 * net['pore.diameter'][np.arange(orig_conduits[conduit, 0], orig_conduits[conduit, 1] + 1)]**2 * conduit_element_length)
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
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns']) # orig_perc_net is needed for calculating measures related to non-functional components (these are removed from perc_net before the simulations)
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
            effective_conductances[time_step], pore_pressures = simulations.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[time_step], functional_susceptibility[time_step], _ = get_conduit_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[time_step::] = net['pore.coords'].shape[0] - time_step
                nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter']**2 * conduit_element_length)
                lcc_size[time_step::] = max_removed_lcc
                prevalence[time_step::] = 1
                time_step += 1
                break
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[time_step::] = net['pore.coords'].shape[0] - time_step
                nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'] * conduit_element_length)
                lcc_size[time_step::] = max_removed_lcc
                prevalence[time_step::] = 1
                time_step += 1
                break
            else:
                raise
        if time_step > si_tolerance_length:
            prevalence_diff = np.abs(prevalence[time_step] - prevalence[time_step - si_tolerance_length])
            
        time_step += 1
    return effective_conductances[0:time_step], lcc_size[0:time_step], functional_lcc_size[0:time_step], nonfunctional_component_size[0:time_step], susceptibility[0:time_step], functional_susceptibility[0:time_step], n_inlets[0:time_step], n_outlets[0:time_step], nonfunctional_component_volume[0:time_step], prevalence[0:time_step]

def run_physiological_conduit_drainage(net, cfg, start_conduits):
    """
    Simulates air drainage in the conduit network assuming that each conduit gets embolized as soon as the pressure difference between air and water phases exceeds conduit's
    invasion pressure. The invasion pressures are calculated using the OpenPNM drainage algorithm, and the effective conductance and a set of network measures is calculated at 
    each pressure.
    
    Parameters:
    -----------
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
        air_contact_angle: float, the contact angle between the water and air phases (degrees)
        surface_tension: float, the surface tension betweent he water and air phases (Newtons/meter)
        inlet_pressure: float, pressure at the inlet conduit elements (Pa)
        outlet_pressure: float, pressure at the outlet conduit elements (Pa) 
        Dp: float, average pit membrane pore diameter (m)
        Tm: float, average thickness of membranes (m)
        conduit_element_length : float, length of a single conduit element (m), used only if use_cylindrical_coords == True (default from the Mrad et al. article)
        heartwood_d : float, diameter of the heartwood (= part of the tree not included in the xylem network) (in conduit elements) used only if use_cylindrical_coords == True (default value from the Mrad et al. article)
        pressure : int or array-like, the frequency steps to investigate (if an int is given, a log-range with the corresponding number of steps is used)
    start_conduits : str or array-like of ints
        the first conduits to be removed (i.e. the first infected node of the simulation)
        if 'random', a single start conduit is selected at random
        if 'bottom', all conduits with pores at the inlet row are used as start conduits
        if an array-like of ints is given, the ints are used as indices of the start conduits

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
    if not cfg['conduit_diameters'] == 'inherit_from_net':
        print("NOTE: pore diameters re-defined in percolation; you may want to set cfg['conduit_diameters'] to 'inherit_from_net' to avoid this")
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
    orig_conduits = conduits.copy()
    
    if start_conduits == 'random':
        start_conduit = np.random.randint(conduits.shape[0])
        start_conduits = [start_conduit]
    elif start_conduits == 'bottom':
        start_conduits = get_inlet_conduits(net, conduits, cec_indicator=cec_indicator, conduit_element_length=conduit_element_length, 
                                            heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
    n_start_conduits = len(start_conduits)
    start_pores = []
    for conduit in start_conduits:
        start_pores.extend(np.arange(conduits[conduit, 0], conduits[conduit, 1] + 1))
    start_pores = np.array(start_pores)
    
    # obtaining the invasion pressure of each pore (i.e. the pressure at which each pore gets embolized) and constructing a pressure range
    sim_net = mrad_model.prepare_simulation_network(net, cfg)
    invasion_pressure = simulations.simulate_drainage(sim_net, start_pores, cfg)
    pressure_difference = np.diff(np.sort(invasion_pressure))
    pressure_step = pressure_difference[pressure_difference > 0].min()
    if np.amax(invasion_pressure) == np.inf:
        max_pressure = np.sort(np.unique(invasion_pressure))[-2]
    else:
        max_pressure = np.amax(invasion_pressure)
    pressure_range = np.arange(0, max_pressure, pressure_step)
    pressure_range_length = pressure_range.shape[0]
    conduit_invasion_pressures = np.zeros((conduits.shape[0], 2))
    conduit_invasion_pressures[:, 1] = 1 # the second column indicates if the conduit is functional (and non-embolized)
    for i, conduit in enumerate(conduits):
        conduit_invasion_pressure = invasion_pressure[np.arange(conduit[0], conduit[1] + 1)]
        assert np.amax(conduit_invasion_pressure) == np.amin(conduit_invasion_pressure), 'Detected a conduit with multiple invasion pressures'
        conduit_invasion_pressures[i, 0] = np.amax(conduit_invasion_pressure)
        
    effective_conductances = np.zeros(pressure_range_length)
    lcc_size = np.zeros(pressure_range_length)
    functional_lcc_size = np.zeros(pressure_range_length)
    nonfunctional_component_size = np.zeros(pressure_range_length)
    nonfunctional_component_volume = np.zeros(pressure_range_length)
    susceptibility = np.zeros(pressure_range_length)
    functional_susceptibility = np.zeros(pressure_range_length)
    n_inlets = np.zeros(pressure_range_length)
    n_outlets = np.zeros(pressure_range_length)
    prevalence = np.zeros(pressure_range_length)
    if not cfg['conduit_diameters'] == 'inherit_from_net':
        cfg['conduit_diameters'] = 'inherit_from_net'
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    perc_net['pore.diameter'] = net['pore.diameter']
    perc_net['pore.invasion_pressure'] = invasion_pressure
    
    max_removed_lcc = 0
    embolized_conduits = []
    n_embolized = 0
    
    n_nonfunctional = 0
    
    for i, pressure in enumerate(pressure_range):
            
        pores_to_remove = []
        
        if i == 0:
            embolized_conduits = np.where(conduit_invasion_pressures[:, 0] <= pressure)[0] # finds all conduits embolized at pressures smaller than the first one investigated
        else:
            # removing start conduits that have gotten embolized or non-functional and re-defining start pores
            j = 0        
            while j < n_start_conduits:
                if conduit_invasion_pressures[start_conduits[j], 1] <= 0:
                    start_conduits.pop(j)
                    n_start_conduits -= 1
                else:
                    j += 1
            start_pores = []
            for conduit in start_conduits:
                start_pores.extend(np.arange(conduits[conduit, 0], conduits[conduit, 1] + 1))
            start_pores = np.array(start_pores)
            
            # recalculating invasion pressures for functional conduits (the pressure of non-functional conduits doesn't change)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            invasion_pressure = simulations.simulate_drainage(sim_net, start_pores, cfg)
            functional_conduits = get_conduit_indices(conduits, np.arange(perc_net['pore.all'].shape[0]))
            for conduit in functional_conduits:
                conduit_invasion_pressures[conduit, 0] = np.amax(invasion_pressure[np.arange(conduits[conduit, 0], conduits[conduit, 1] + 1)])
            
            embolized_conduits = np.where((conduit_invasion_pressures[:, 0] <= pressure) & (conduit_invasion_pressures[:, 0] > pressure_range[i - 1]))[0] # conduits embolized at this pressure but not at the previous pressure
        n_embolized += embolized_conduits.shape[0]
        for embolized_conduit in np.sort(embolized_conduits)[::-1]:
            if conduit_invasion_pressures[embolized_conduit, 1] > 0:
                pores_to_remove.extend(list(np.arange(conduits[embolized_conduit, 0], conduits[embolized_conduit, 1] + 1)))
                conduits[embolized_conduit + 1::, 0:2] = conduits[embolized_conduit + 1::, 0:2] - conduits[embolized_conduit, 2]
                conduits[embolized_conduit, :] = -1
                conduit_invasion_pressures[embolized_conduit, 1] = 0
            else: # if a nonfunctional conduit is embolized, nonfunctional component size and volume decrease
                nonfunctional_component_size[i] -= 1
                nonfunctional_component_volume[i] -= np.sum(np.pi * 0.5 * net['pore.diameter'][np.arange(orig_conduits[embolized_conduit, 0], orig_conduits[embolized_conduit, 1] + 1)]**2 * conduit_element_length)
        op.topotools.trim(perc_net, pores=pores_to_remove)
        prevalence[i] = n_embolized / conduits.shape[0]
    
        try:
            lcc_size[i], susceptibility[i], _ = get_conduit_lcc_size(perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns']) # orig_perc_net is needed for calculating measures related to non-functional components (these are removed from perc_net before the simulations)
            orig_perc_net['throat.type'] = perc_net['throat.type']
            orig_perc_net['pore.diameter'] = perc_net['pore.diameter']
            perc_net, removed_components = mrad_model.clean_network(perc_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False) # removes non-functional components from perc_net
            removed_elements = list(itertools.chain.from_iterable(removed_components)) # calculating the size of the largest removed component in conduits
            nonfunctional_component_volume[i] += nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'][removed_elements]**2 * conduit_element_length)
            if len(removed_elements) > 0:
                removed_net = get_induced_subnet(orig_perc_net, removed_elements)
                removed_lcc, _, n_nonfunctional_conduits = get_conduit_lcc_size(removed_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                       conduit_element_length=conduit_element_length, 
                                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
                nonfunctional_component_size[i] = nonfunctional_component_size[i] + nonfunctional_component_size[i - 1] + n_nonfunctional_conduits
                if removed_lcc > max_removed_lcc: # Percolation doesn't affect the sizes of removed components -> the largest removed component size changes only if a new, larger component gets removed
                    max_removed_lcc = removed_lcc
                removed_conduits = get_conduit_indices(conduits, removed_elements)
                for removed_conduit in removed_conduits[::-1]: # indexing of conduits starts from 1, not 0
                    conduits[removed_conduit + 1::, 0:2] = conduits[removed_conduit + 1::, 0:2] - conduits[removed_conduit, 2]
                    conduits[removed_conduit, :] = -1
                conduit_invasion_pressures[removed_conduits, 1] = 0
                
                n_nonfunctional += removed_conduits.shape[0]
                
            else:
                nonfunctional_component_size[i] = nonfunctional_component_size[i] + nonfunctional_component_size[i - 1]
            if max_removed_lcc > lcc_size[i]: # checking if the largest removed component is larger than the largest functional one
                lcc_size[i] = max_removed_lcc
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, cfg['net_size'][0] - 1, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            sim_net = mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[i], _ = simulations.simulate_water_flow(sim_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i], _ = get_conduit_lcc_size(perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes belong to non-functional components
                nonfunctional_component_size[i::] = net['pore.coords'].shape[0] - embolized_conduits.shape[0]
                nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter']**2 * conduit_element_length)
                lcc_size[i::] = max_removed_lcc
                prevalence[i::] = 1
                break
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[i::] = net['pore.coords'].shape[0] - embolized_conduits.shape[0]
                nonfunctional_component_volume[i::] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'] * conduit_element_length)
                lcc_size[i::] = max_removed_lcc
                prevalence[i::] = 1
                break
            else:
                raise
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence

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
            start_conduit = conduit_elements[throat[0], 3]
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
        start_conduit = int(conduit_elements[throat[0], 3])
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
    Finds the indices of the conduits (sets of consecutive elements in the row/z direction), which given elements belong to.

    Parameters
    ----------
    conduits : np.array
        three columns: 1) the first element of the conduit, 2) the last element of the conduit, 3) size of the conduit
    elements : iterable of ints
        list of elements; each element should belong to one of the conduits 

    Returns
    -------
    conduit_indices : np.array
        the indices of conduits, to which the elements belong. Note that each conduit index is included only once even if
        there are multiple elements from this conduit, and conduit_indices.shape doesn't match elements.shape.
    """
    conduit_indices = []
    for element in elements:
        conduit_index = np.where((conduits[:, 0] <= element) & (conduits[:, 1] >= element))[0][0]
        conduit_indices.append(conduit_index)
    conduit_indices = np.unique(conduit_indices)
    return conduit_indices
            
def get_inlet_conduits(net, conduits, cec_indicator=params.cec_indicator, conduit_element_length=params.Lce, 
                 heartwood_d=params.heartwood_d, use_cylindrical_coords=False):
    """
    Finds the conduits that contain inlet pores (pores at the bottom row)

    Parameters
    ----------
    net : openpnm.Network()
        a network object
    conduits : np.array
        three columns: 1) the first element of the conduit, 2) the last element of the conduit, 3) size of the conduit
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
    inlet_conduit_indices : list
        indices of the conduits with inlet pores
    """
    conduit_elements = mrad_model.get_conduit_elements(net, use_cylindrical_coords=use_cylindrical_coords,
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    inlet_conduit_indices = []
    for pore in conduit_elements:
        if pore[0] == 0:
            inlet_conduit_indices.append(int(pore[3]))
    return inlet_conduit_indices

def calculate_bpp(net, conduits, icc_mask, cfg):
    """
    Calculates the bubble propagation pressure (BPP) for each ICC as 4*gamma/D_p where D_p is drawn from Weibull
    distribution (equation 6 of the Mrad et al. article). 

    Parameters
    ----------
    net : openpnm.network()
        pores correspond to conduit elements, throats to CECs and ICCs
    conduits : np.array
        three columns: 1) the first element of the conduit, 2) the last element of the conduit, 3) size of the conduit
    icc_mask : np.array
        for each throat of the network, contains 1 if the throat is an ICC and 0 otherwise
    cfg : dict
        contains: 
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6)
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2)
        average_pit_area : float, the average area of a pit
        Lce: float, length of a conduit element
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits
    Returns
    -------
    bpp : np.array
        bubble propagation pressure for each ICC of the network
    """
    weibull_a = cfg.get('weibull_a', params.weibull_a)
    weibull_b = cfg.get('weibull_b', params.weibull_b)
    average_pit_area = cfg.get('average_pit_area', params.Dm**2)
    Dc = cfg.get('Dc', params.Dc)
    Dc_cv = cfg.get('Dc_cv', params.Dc_cv)
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    fc = cfg.get('fc', params.fc)
    fpf = cfg.get('fpf', params.fpf)
    conns = net['throat.conns']
    
    diameters_per_conduit, _ = mrad_model.get_conduit_diameters(net, 'inherit_from_net', conduits, Dc_cv=Dc_cv, Dc=Dc)
    conduit_areas = (conduits[:, 2] - 1) * conduit_element_length * np.pi * diameters_per_conduit
    iccs = conns[np.where(icc_mask)]
    icc_count = np.array([np.sum((conduit[0] <= iccs[:, 0]) & (iccs[:, 0] <= conduit[1])) + np.sum((conduit[0] <= iccs[:, 1]) & (iccs[:, 1] <= conduit[1])) for conduit in conduits])
    
    bpp = np.zeros(iccs.shape[0])
    for i, icc in enumerate(iccs):
        start_conduit = np.where((conduits[:, 0] <= icc[0]) & (icc[0] <= conduits[:, 1]))[0][0]
        end_conduit = np.where((conduits[:, 0] <= icc[1]) & (icc[1] <= conduits[:, 1]))[0][0]
        Am = 0.5 * (conduit_areas[start_conduit] / icc_count[start_conduit] + conduit_areas[end_conduit] / icc_count[end_conduit]) * fc * fpf
        pit_count = Am / average_pit_area
        bpp[i] = (weibull_a / pit_count**(1 / weibull_b)) * np.random.weibull(weibull_b)
        
    return bpp