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
import pickle
import os

import mrad_model
import mrad_params as params
import simulations
import bubble_expansion
import pit_membrane

def run_percolation(net, cfg, percolation_type='bond', removal_order='random', break_nonfunctional_components=True, include_orig_values=False):
    """
    Removes throats (bond percolation) or pores (site percolation) from an OpenPNM network object in a given (or random) order and calculates 
    a number of measures, including effective conductance and largest component size, after removal (see below for details). If order is given,
    percolation is "graph-theoretical", i.e. nonfunctional components (i.e. components not connected to inlet or outlet) can
    be further broken. Percolation with random order can be also "physiological", i.e. nonfunctional components are not
    broken further.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between the elements. Note that the Network object needs to be prepared for running simulations
        by mrad_model.prepare_simulation_network() or otherwise; this preparation includes setting the geometry of throats and pores.
    cfg : dict
        contains:
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
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
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading); only used if bpp_type == 'young-laplace'
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading); only used if bpp_type == 'young-laplace'
        bpp_type: str, how the bubble propagation pressure is calculated; options: 'young-laplace' (i.e. as in Mrad et al. 2018) and 'young-laplace_with_constrictions' (i.e. as in Kaack et al. 2021)
        bpp_data_path : str, optional, path, to which the BPP data has been saved; only used if bpp_type == 'young-laplace_with_constrictions'
    percolation type : str, optional
        type of the percolation, 'bond' to remove throats, 'site' to remove pores, 'conduit' to remove conduits,
        'si' to simulate the spreading of embolism between conduits using the SI model or 'drainage' to simulate the spreading of embolism with the openpnm drainage algorithm (default: 'bond')
    removal_order : iterable or str, optional (default='random')
        indices of pores to be removed in the order of removal. for performing full percolation analysis, len(removal_order)
        should match the number of pores. string 'random' can be given to indicate removal in random order.
    break_nonfunctional_components : bln, optional
        can links/nodes in components that are nonfunctional (i.e. not connected to inlet or outlet) be removed (default: True)
    include_orig_values : bln, optional
        should the output arrays include also values calculated for the intact network in addition to the values after each removal (default: False)

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
    prevalence_due_to_spontaneous_embolism : np.array
        the fraction of conduits embolized spontaneously (only calculated if percolation_type == 'si')
    prevalence_due_to_spreading : np.array
        the fraction of conduits embolized through embolism spreading (only calculated if percolation_type == 'si')
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
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading = run_conduit_si(net, cfg, spreading_param, include_orig_values)
    elif percolation_type == 'drainage':
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence = run_physiological_conduit_drainage(net, cfg, start_conduits=cfg['start_conduits'])
    elif break_nonfunctional_components:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_graph_theoretical_element_percolation(net, cfg, percolation_type, removal_order)
    else:
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume = run_physiological_element_percolation(net, cfg, percolation_type)
    if not percolation_type in ['si', 'drainage']:
        prevalence = np.zeros(len(effective_conductances))
        prevalence_due_to_spontaneous_embolism = np.zeros(len(effective_conductances))
        prevalence_due_to_spreading = np.zeros(len(effective_conductances))
    
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading

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
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6); only used if bpp_type == 'young-laplace'
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2); only used if bpp_type == 'young-laplace'
        average_pit_area : float, the average area of a pit
        nCPUs : int, number of CPUs used for parallel computing (default 5)
        bpp_type: str, how the bubble propagation pressure is calculated; options: 'young-laplace' (i.e. as in Mrad et al. 2018) and 'young-laplace_with_constrictions' (i.e. as in Kaack et al. 2021)
        bpp_data_path : str, optional, path, to which the BPP data has been saved; only used if bpp_type == 'young-laplace_with_constrictions'
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
    si_type = cfg.get('si_type', 'stochastic')
    if si_type == 'physiological' and x_range[0] > 0:
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
    if si_type == 'physiological':
        vulnerability = 100 * (1 - vulnerability / vulnerability[0]) # normalization to obtain the percentage of conductance lost compared to the first x_range value
    else: # si_type == 'stochastic'
        vulnerability = 100 * (1 - vulnerability / output[0][0][0]) # normalization by the effective conductance of the intact network
    vc = (x_range, vulnerability)
    return vc

def optimize_spreading_probability(net, cfg, pressure_difference, spreading_probability_range=np.arange(0.001, 1, step=0.1), si_length=1000, n_iterations=1, save_path_base=''):
    """
    Finds the SI spreading probability that yields final effective conductance as close as possible to that of physiological embolism spreading with given parameters. Note that
    only the similarity of final effective conductances is minimized, while the shape of the prevalence curves may be different.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between the elements
    cfg : dict
        contains:
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
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
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6); only used if bpp_type == 'young-laplace'
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2); only used if bpp_type == 'young-laplace'
        average_pit_area : float, the average area of a pit
        nCPUs : int, number of CPUs used for parallel computing (default 5)
        spontaneous_embolism : bln, is spontaneous embolism through bubble expansion allowed (default: False)
        spontaneous_embolism_probabilities : dic where keys are pressure values and values probabilities for spontaneous embolism
        start_conduits : str or array-like of ints
           the first conduits to be removed (i.e. the first infected node of the simulation)
           if 'random', a single start conduit is selected at random
           if 'random_per_component', a single start conduit per network component is selected at random
           if an array-like of ints is given, the ints are used as indices of the start conduits
        bpp_type: str, how the bubble propagation pressure is calculated; options: 'young-laplace' (i.e. as in Mrad et al. 2018) and 'young-laplace_with_constrictions' (i.e. as in Kaack et al. 2021)
        bpp_data_path : str, optional, path, to which the BPP data has been saved; only used if bpp_type == 'young-laplace_with_constrictions'
    pressure_difference : float
        pressure difference between water in conduits and air (bubble)
    spreading_probability_range : array-like
        the spreading probability values, among which the optimal one is selected
    si_length : int, optional
        maximum number of time steps used for the simulation. The default is 1000.
    n_iterations : int, optional
        number of iterations with different random seeds. The default is 1.
    save_path_base : str, optional
        base path to which to save the pressure difference, optimal spreading probability and the effective conductance values corresponding to these. if no save_path_base is
        given, these values are returned instead.

    Returns
    -------
    if save_path_base is given, None
    else output : dict
         contains:
         pressured_difference : float
         optimal_spreading_probability : float, the spreading probability yielding the effective conductance closest to the physiological value
         physiological_effective_conductance : float, the effective conductance at the end of the physiological embolism spreading
         stochastic_effective_conductance : float, the effective conductance at the end of the stochastic embolism spreading
    """
    cfg['si_length'] = si_length
    cfg['si_type'] = 'physiological'
    physiological_effective_conductances = np.zeros(n_iterations)
    stochastic_effective_conductances = np.zeros((len(spreading_probability_range), n_iterations))

    spontaneous_embolism = cfg.get('spontaneous_embolism', False)
    if spontaneous_embolism:
        cfg['spontaneous_embolism_probability'] = cfg['spontaneous_embolism_probabilities'][pressure_difference]
    
    for i in np.arange(n_iterations):
        physiological_effective_conductances[i] = run_conduit_si(net, cfg, pressure_difference)[0][-1]
    physiological_effective_conductance = np.mean(physiological_effective_conductances)
    
    cfg['si_type'] = 'stochastic'
    for i, spreading_probability in enumerate(spreading_probability_range):
        for j in np.arange(n_iterations):
            stochastic_effective_conductances[i, j] = run_conduit_si(net, cfg, spreading_probability)[0][-1]
    
    stochastic_effective_conductances = np.mean(stochastic_effective_conductances, axis=1)
    
    optimal_spreading_probability_index = np.argmin(np.abs(stochastic_effective_conductances - physiological_effective_conductance))
    optimal_spreading_probability = spreading_probability_range[optimal_spreading_probability_index]
    stochastic_effective_conductance = stochastic_effective_conductances[optimal_spreading_probability_index]
    
    output = {'pressure_difference':pressure_difference, 'optimized_spreading_probability': optimal_spreading_probability, 'physiological_effective_conductance': physiological_effective_conductance, 'stochastic_effective_conductance': stochastic_effective_conductance}
    
    if len(save_path_base) > 0:
        save_folder = save_path_base.rsplit('/', 1)[0]
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = save_path_base + '_' + str(pressure_difference).replace('.', '_') + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(output, f)
        f.close()
    else:
        return output
    
def run_spreading_iteration(net, cfg, pressure_differences, save_path, spreading_probability_range=np.arange(0.001, 1, step=0.1), si_length=1000, include_orig_values=False):
    """
    Runs physiological conduit SI for a range of pressure difference values and stochastic conduit SI for a range of spreading probability values and saves the effective conductance
    value and prevalence curve of each simulation. Used for creating the data for optimizing spreading probability.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between the elements
    cfg : dict
        contains:
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
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
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6); only used if bpp_type == 'young-laplace'
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2); only used if bpp_type == 'young-laplace'
        average_pit_area : float, the average area of a pit
        spontaneous_embolism : bln, is spontaneous embolism through bubble expansion allowed (default: False)
        spontaneous_embolism_probabilities : dic where keys are pressure values and values probabilities for spontaneous embolism
        start_conduits : str or array-like of ints
            the first conduits to be removed (i.e. the first infected node of the simulation)
            if 'random', a single start conduit is selected at random
            if 'random_per_component', a single start conduit per network component is selected at random
            if an array-like of ints is given, the ints are used as indices of the start conduits
        bpp_type: str, how the bubble propagation pressure is calculated; options: 'young-laplace' (i.e. as in Mrad et al. 2018) and 'young-laplace_with_constrictions' (i.e. as in Kaack et al. 2021)
        bpp_data_path : str, optional, path, to which the BPP data has been saved; only used if bpp_type == 'young-laplace_with_constrictions'
        segment_name ; str, optional, name of the segment to be analyzed; if this is given, it's saved along the simulation outcomes
    pressure_differences : iterable of float
        pressure differences between water in conduits and air (bubble), with which the physiological SI is simulated
    save_path : str
        base path to which to save the outcomes
    spreading_probability_range : array-like, optional
        the spreading probability values, with which the stochastic SI is simulated
    si_length : int, optional
        maximum number of time steps used for the simulation. The default is 1000.
    include_orig_values : bln, optional
        should the saved arrays include also values calculated for the intact network in addition to the values after each removal (default: False)

    Returns
    -------
    None
    """
    spontaneous_embolism = cfg.get('spontaneous_embolism', False)
    physiological_effective_conductances = np.zeros(len(pressure_differences))
    physiological_full_effective_conductances = []
    physiological_prevalences = []
    physiological_prevalences_due_to_spontaneous_embolism = []
    physiological_prevalences_due_to_spreading = []
    physiological_lcc_size = []
    physiological_functional_lcc_size = []
    physiological_nonfunctional_component_size = []
    physiological_nonfunctional_component_volume = []
    physiological_susceptibility = []
    physiological_functional_susceptibility = []
    physiological_n_inlets = []
    physiological_n_outlets = []
    if spontaneous_embolism:
        spontaneous_embolism_probabilities = cfg['spontaneous_embolism_probabilities']
        spontaneous_embolism_pressure_differences = np.sort(np.array(list(spontaneous_embolism_probabilities.keys())))
        stochastic_effective_conductances = np.zeros((len(spreading_probability_range), len(spontaneous_embolism_pressure_differences)))
        stochastic_full_effective_conductances = [[] for spreading_probability in spreading_probability_range]
        stochastic_prevalences = [[] for spreading_probability in spreading_probability_range]
        stochastic_prevalences_due_to_spontaneous_embolism = [[] for spreading_probability in spreading_probability_range]
        stochastic_prevalences_due_to_spreading = [[] for spreading_probability in spreading_probability_range]
        stochastic_lcc_size = [[] for spreading_probability in spreading_probability_range]
        stochastic_functional_lcc_size = [[] for spreading_probability in spreading_probability_range]
        stochastic_nonfunctional_component_size = [[] for spreading_probability in spreading_probability_range]
        stochastic_nonfunctional_component_volume = [[] for spreading_probability in spreading_probability_range]
        stochastic_susceptibility = [[] for spreading_probability in spreading_probability_range]
        stochastic_functional_susceptibility = [[] for spreading_probability in spreading_probability_range]
        stochastic_n_inlets = [[] for spreading_probability in spreading_probability_range]
        stochastic_n_outlets = [[] for spreading_probability in spreading_probability_range]
    else:
        spontaneous_embolism_pressure_differences = []
        stochastic_effective_conductances = np.zeros(len(spreading_probability_range))
        stochastic_full_effective_conductances = []
        stochastic_prevalences = []
        stochastic_prevalences_due_to_spontaneous_embolism = []
        stochastic_prevalences_due_to_spreading = []
        stochastic_lcc_size = []
        stochastic_functional_lcc_size = []
        stochastic_nonfunctional_component_size = []
        stochastic_nonfunctional_component_volume = []
        stochastic_susceptibility = []
        stochastic_functional_susceptibility = []
        stochastic_n_inlets = []
        stochastic_n_outlets = []
    
    # running physiological SI
    cfg['si_type'] = 'physiological'
    for i, pressure_difference in enumerate(pressure_differences):
        if spontaneous_embolism:
            cfg['spontaneous_embolism_probability'] = spontaneous_embolism_probabilities[pressure_difference]
        effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading = run_conduit_si(net, cfg, pressure_difference, include_orig_values)
        physiological_effective_conductances[i] = effective_conductances[-1]
        physiological_full_effective_conductances.append(effective_conductances)
        physiological_prevalences.append(prevalence)
        physiological_prevalences_due_to_spontaneous_embolism.append(prevalence_due_to_spontaneous_embolism)
        physiological_prevalences_due_to_spreading.append(prevalence_due_to_spreading)
        physiological_lcc_size.append(lcc_size)
        physiological_functional_lcc_size.append(functional_lcc_size)
        physiological_nonfunctional_component_size.append(nonfunctional_component_size)
        physiological_nonfunctional_component_volume.append(nonfunctional_component_volume)
        physiological_susceptibility.append(susceptibility)
        physiological_functional_susceptibility.append(functional_susceptibility)
        physiological_n_inlets.append(n_inlets)
        physiological_n_outlets.append(n_outlets)
        
    # running stochastic SI
    cfg['si_type'] = 'stochastic'
    if spontaneous_embolism:
        for i, spreading_probability in enumerate(spreading_probability_range):
            for j, pressure_difference in enumerate(spontaneous_embolism_pressure_differences):
                cfg['spontaneous_embolism_probability'] = spontaneous_embolism_probabilities[pressure_difference]
                effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading = run_conduit_si(net, cfg, spreading_probability, include_orig_values)
                stochastic_effective_conductances[i, j] = effective_conductances[-1]
                stochastic_full_effective_conductances[i].append(effective_conductances)
                stochastic_prevalences[i].append(prevalence)
                stochastic_prevalences_due_to_spontaneous_embolism[i].append(prevalence_due_to_spontaneous_embolism)
                stochastic_prevalences_due_to_spreading[i].append(prevalence_due_to_spreading)
                stochastic_lcc_size[i].append(lcc_size)
                stochastic_functional_lcc_size[i].append(functional_lcc_size)
                stochastic_nonfunctional_component_size[i].append(nonfunctional_component_size)
                stochastic_nonfunctional_component_volume[i].append(nonfunctional_component_volume)
                stochastic_susceptibility[i].append(susceptibility)
                stochastic_functional_susceptibility[i].append(functional_susceptibility)
                stochastic_n_inlets[i].append(n_inlets)
                stochastic_n_outlets[i].append(n_outlets)
    else:
        for i, spreading_probability in enumerate(spreading_probability_range):
            effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading = run_conduit_si(net, cfg, spreading_probability, include_orig_values)
            stochastic_effective_conductances[i] = effective_conductances[-1]
            stochastic_full_effective_conductances.append(effective_conductances)
            stochastic_prevalences.append(prevalence)
            stochastic_prevalences_due_to_spontaneous_embolism.append(prevalence_due_to_spontaneous_embolism)
            stochastic_prevalences_due_to_spreading.append(prevalence_due_to_spreading)
            stochastic_lcc_size.append(lcc_size)
            stochastic_functional_lcc_size.append(functional_lcc_size)
            stochastic_nonfunctional_component_size.append(nonfunctional_component_size)
            stochastic_nonfunctional_component_volume.append(nonfunctional_component_volume)
            stochastic_susceptibility.append(susceptibility)
            stochastic_functional_susceptibility.append(functional_susceptibility)
            stochastic_n_inlets.append(n_inlets)
            stochastic_n_outlets.append(n_outlets)
     
    # saving simulation outputs
    data = {'pressure_differences': pressure_differences, 'spreading_probability_range': spreading_probability_range, 'spontaneous_embolism_pressure_differences': spontaneous_embolism_pressure_differences, 'physiological_effective_conductances': physiological_effective_conductances,
            'physiological_prevalences': physiological_prevalences, 'stochastic_effective_conductances': stochastic_effective_conductances, 'stochastic_prevalences': stochastic_prevalences,
            'spontaneous_embolism': spontaneous_embolism, 'physiological_prevalences_due_to_spontaneous_embolism': physiological_prevalences_due_to_spontaneous_embolism,
            'physiological_prevalences_due_to_spreading': physiological_prevalences_due_to_spreading, 'stochastic_prevalences_due_to_spontaneous_embolism': stochastic_prevalences_due_to_spontaneous_embolism,
            'stochastic_prevalences_due_to_spreading': stochastic_prevalences_due_to_spreading, 'physiological_lcc_size': physiological_lcc_size, 'physiological_functional_lcc_size': physiological_functional_lcc_size,
            'physiological_nonfunctional_component_size': physiological_nonfunctional_component_size, 'physiological_nonfunctional_component_volume': physiological_nonfunctional_component_volume, 'physiological_susceptibility': physiological_susceptibility,
            'physiological_functional_susceptibility': physiological_functional_susceptibility, 'physiological_n_inlets': physiological_n_inlets,
            'physiological_n_outlets': physiological_n_outlets, 'stochastic_lcc_size': stochastic_lcc_size, 'stochastic_functional_lcc_size': stochastic_functional_lcc_size, 'stochastic_nonfunctional_component_size': stochastic_nonfunctional_component_size,
            'stochastic_nonfunctional_component_volume': stochastic_nonfunctional_component_volume, 'stochastic_susceptibility': stochastic_susceptibility, 'stochastic_functional_susceptibility': stochastic_functional_susceptibility,
            'stochastic_n_inlets': stochastic_n_inlets, 'stochastic_n_outlets': stochastic_n_outlets, 'physiological_full_effective_conductances': physiological_full_effective_conductances, 'stochastic_full_effective_conductances': stochastic_full_effective_conductances, 'cfg':cfg}
    if 'segment_name' in cfg.keys():
        data['segment_name'] = cfg['segment_name']
    save_folder = save_path.rsplit('/', 1)[0]
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
    
def optimize_spreading_probability_from_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, equal_spreading_params=False, max_n_iterations=None, pool_physiological_only=False):
    """
    Starting from previously calculated simulation data, finds the SI spreading probability that yields final effective conductance as close as possible to that of 
    physiological embolism spreading with given parameters. Note that only the similarity of final effective conductances is minimized, while the shape of the prevalence curves may be different.

    Parameters
    ----------
    simulation_data_save_folder : str
        path of the folder in which the simulation data is saved
    simulation_data_save_name_base : str
        stem of the simulation data file names; the file names can contain other parts but files with names that don't contain this stem aren't read
    pooled_data_save_path : str
        path to which to save the pressure difference, optimal spreading probability and the effective conductance values corresponding to these should be saved
    equal_spreading_params : bln, optional
        do all data files contain exactly the same pressure difference range and spreading probability range (this affects how the data is read) (default: False)
    max_n_iterations : int, optional
        the maximum number of iterations read per pressure difference or spreading probability (default None, in which case all iterations available are read)
    pool_physiological_only : bln, optional
        if True, physiological data is pooled and saved ina format that allows constructing a vulnerability curve but spreading probability is not optimized
        (this option can be used e.g. if there is no stochastic data) (default: False)

    Returns
    -------
    None.
    """   
    if equal_spreading_params:
        data_files = [os.path.join(simulation_data_save_folder, file) for file in os.listdir(simulation_data_save_folder) if os.path.isfile(os.path.join(simulation_data_save_folder, file))]
        data_files = [data_file for data_file in data_files if simulation_data_save_name_base in data_file]
        if pooled_data_save_path in data_files:
            data_files.remove(pooled_data_save_path)
        realized_n_pressure_iterations = len(data_files)
        realized_n_probability_iterations = len(data_files)
        for i, data_file in enumerate(data_files):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                f.close()
            spontaneous_embolism = data['spontaneous_embolism']
            if i == 0:
                pressure_differences = data['pressure_differences']
                spreading_probability_range = data['spreading_probability_range']
                physiological_effective_conductances = np.zeros((len(pressure_differences), len(data_files))) # pressures x iterations
                physiological_full_effective_conductances = []
                physiological_prevalences = []
                physiological_prevalences_due_to_spontaneous_embolism = []
                physiological_prevalences_due_to_spreading = []
                physiological_lcc_size = []
                physiological_functional_lcc_size = []
                physiological_nonfunctional_component_size = []
                physiological_nonfunctional_component_volume = []
                physiological_susceptibility = []
                physiological_functional_susceptibility = []
                physiological_n_inlets = []
                physiological_n_outlets = []
                if spontaneous_embolism:
                    stochastic_effective_conductances = np.zeros((len(spreading_probability_range), len(pressure_differences), len(data_files))) # spreading probabilities x pressures x iterations
                else:
                    stochastic_effective_conductances = np.zeros((len(spreading_probability_range), len(data_files))) # spreading probabilities x iterations
                stochastic_full_effective_conductances = []
                stochastic_prevalences = []
                stochastic_prevalences_due_to_spontaneous_embolism = []
                stochastic_prevalences_due_to_spreading = []
                stochastic_lcc_size = []
                stochastic_functional_lcc_size = []
                stochastic_nonfunctional_component_size = []
                stochastic_nonfunctional_component_volume = []
                stochastic_susceptibility = []
                stochastic_functional_susceptibility = []
                stochastic_n_inlets = []
                stochastic_n_outlets = []
            else:
                assert np.all(data['pressure_differences'] == pressure_differences), data_file + ' has different list of pressure_differences than other files'
                assert np.all(data['spreading_probability_range'] == spreading_probability_range), data_file + ' has different spreading probability range than other files'
            physiological_effective_conductances[:, i] = data['physiological_effective_conductances']
            physiological_full_effective_conductances.append(data['physiological_full_effective_conductances'])
            physiological_prevalences.append(data['physiological_prevalences'])
            physiological_prevalences_due_to_spontaneous_embolism.append(data['physiological_prevalences_due_to_spontaneous_embolism'])
            physiological_prevalences_due_to_spreading.append(data['physiological_prevalences_due_to_spreading'])
            physiological_lcc_size.append(data['physiological_lcc_size'])
            physiological_functional_lcc_size.append(data['physiological_functional_lcc_size'])
            physiological_nonfunctional_component_size.append(data['physiological_nonfunctional_component_size'])
            physiological_nonfunctional_component_volume.append(data['physiological_nonfunctional_component_volume'])
            physiological_susceptibility.append(data['physiological_susceptibility'])
            physiological_functional_susceptibility.append(data['physiological_functional_susceptibility'])
            physiological_n_inlets.append(data['physiological_n_inlets'])
            physiological_n_outlets.append(data['physiological_n_outlets'])
            if spontaneous_embolism:
                stochastic_effective_conductances[:, :, i] = data['stochastic_effective_conductances']
            else:
                stochastic_effective_conductances[:, i] = data['stochastic_effective_conductances']
            stochastic_full_effective_conductances.append(data['stochastic_full_effective_conductances'])
            stochastic_prevalences.append(data['stochastic_prevalences'])
            stochastic_prevalences_due_to_spontaneous_embolism.append(data['stochastic_prevalences_due_to_spontaneous_embolism'])
            stochastic_prevalences_due_to_spreading.append(data['stochastic_prevalences_due_to_spreading'])
            stochastic_lcc_size.append(data['stochastic_lcc_size'])
            stochastic_functional_lcc_size.append(data['stochastic_functional_lcc_size'])
            stochastic_nonfunctional_component_size.append(data['stochastic_nonfunctional_component_size'])
            stochastic_nonfunctional_component_volume.append(data['stochastic_nonfunctional_component_volume'])
            stochastic_susceptibility.append(data['stochastic_susceptibility'])
            stochastic_functional_susceptibility.append(data['stochastic_functional_susceptibility'])
            stochastic_n_inlets.append(data['stochastic_n_inlets'])
            stochastic_n_outlets.append(data['stochastic_n_outlets'])   
    else:
        pressure_differences, spreading_probability_range, physiological_effective_conductances, physiological_full_effective_conductances, physiological_prevalences, physiological_prevalences_due_to_spontaneous_embolism, \
        physiological_prevalences_due_to_spreading, physiological_lcc_size, physiological_functional_lcc_size, physiological_nonfunctional_component_size, physiological_nonfunctional_component_volume, physiological_susceptibility, \
        physiological_functional_susceptibility, physiological_n_inlets, physiological_n_outlets, stochastic_effective_conductances, stochastic_full_effective_conductances, stochastic_prevalences, \
        stochastic_prevalences_due_to_spontaneous_embolism, stochastic_prevalences_due_to_spreading, stochastic_lcc_size, stochastic_functional_lcc_size, stochastic_nonfunctional_component_size, stochastic_nonfunctional_component_volume, \
        stochastic_susceptibility, stochastic_functional_susceptibility, stochastic_n_inlets, stochastic_n_outlets, spontaneous_embolism, realized_n_pressure_iterations, \
        realized_n_probability_iterations,_ = read_and_combine_spreading_probability_optimization_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, max_n_iterations=max_n_iterations)
    
    averaged_physiological_effective_conductances = np.zeros(len(pressure_differences))
    optimized_spreading_probabilities = np.zeros(len(pressure_differences))
    optimized_stochastic_effective_conductances = np.zeros(len(pressure_differences))
    average_phys_full_effective_conductances = []
    std_phys_full_effective_conductances = []
    average_stoch_full_effective_conductances = []
    std_stoch_full_effective_conductances = []
    average_phys_prevalences = []
    std_phys_prevalences = []
    average_phys_prevalences_due_to_spontaneous_embolism = []
    std_phys_prevalences_due_to_spontaneous_embolism = []
    average_phys_prevalences_due_to_spreading = []
    std_phys_prevalences_due_to_spreading = []
    average_stoch_prevalences = []
    std_stoch_prevalences = []
    average_stoch_prevalences_due_to_spontaneous_embolism = []
    std_stoch_prevalences_due_to_spontaneous_embolism = []
    average_stoch_prevalences_due_to_spreading = []
    std_stoch_prevalences_due_to_spreading = []
    av_phys_lcc_size = []
    std_phys_lcc_size = []
    av_stoch_lcc_size = []
    std_stoch_lcc_size = []
    av_phys_functional_lcc_size = []
    std_phys_functional_lcc_size = []
    av_stoch_functional_lcc_size = []
    std_stoch_functional_lcc_size = []
    av_phys_nonfunctional_component_size = []
    std_phys_nonfunctional_component_size = []
    av_stoch_nonfunctional_component_size = []
    std_stoch_nonfunctional_component_size = []
    av_phys_nonfunctional_component_volume = []
    std_phys_nonfunctional_component_volume = []
    av_stoch_nonfunctional_component_volume = []
    std_stoch_nonfunctional_component_volume = []
    av_phys_susceptibility = []
    std_phys_susceptibility = []
    av_stoch_susceptibility = []
    std_stoch_susceptibility = []
    av_phys_functional_susceptibility = []
    std_phys_functional_susceptibility = []
    av_stoch_functional_susceptibility = []
    std_stoch_functional_susceptibility = []
    av_phys_n_inlets = []
    std_phys_n_inlets = []
    av_stoch_n_inlets = []
    std_stoch_n_inlets = []
    av_phys_n_outlets = []
    std_phys_n_outlets = []
    av_stoch_n_outlets = []
    std_stoch_n_outlets = []
    
    phys_props = [physiological_full_effective_conductances, physiological_lcc_size, physiological_functional_lcc_size, physiological_nonfunctional_component_size, physiological_nonfunctional_component_volume, physiological_susceptibility, physiological_functional_susceptibility, physiological_n_inlets, physiological_n_outlets]
    phys_avs = [average_phys_full_effective_conductances, av_phys_lcc_size, av_phys_functional_lcc_size, av_phys_nonfunctional_component_size, av_phys_nonfunctional_component_volume, av_phys_susceptibility, av_phys_functional_susceptibility, av_phys_n_inlets, av_phys_n_outlets]
    phys_stds = [std_phys_full_effective_conductances, std_phys_lcc_size, std_phys_functional_lcc_size, std_phys_nonfunctional_component_size, std_phys_nonfunctional_component_volume, std_phys_susceptibility, std_phys_functional_susceptibility, std_phys_n_inlets, std_phys_n_outlets]
    stoch_props = [stochastic_full_effective_conductances, stochastic_lcc_size, stochastic_functional_lcc_size, stochastic_nonfunctional_component_size, stochastic_nonfunctional_component_volume, stochastic_susceptibility, stochastic_functional_susceptibility, stochastic_n_inlets, stochastic_n_outlets]
    stoch_avs = [average_stoch_full_effective_conductances, av_stoch_lcc_size, av_stoch_functional_lcc_size, av_stoch_nonfunctional_component_size, av_stoch_nonfunctional_component_volume, av_stoch_susceptibility, av_stoch_functional_susceptibility, av_stoch_n_inlets, av_stoch_n_outlets]
    stoch_stds = [std_stoch_full_effective_conductances, std_stoch_lcc_size, std_stoch_functional_lcc_size, std_stoch_nonfunctional_component_size, std_stoch_nonfunctional_component_volume, std_stoch_susceptibility, std_stoch_functional_susceptibility, std_stoch_n_inlets, std_stoch_n_outlets]
    
    for i, pressure_difference in enumerate(pressure_differences):
        n_pressure_iterations = realized_n_pressure_iterations[i]
        physiological_effective_conductance = np.mean(physiological_effective_conductances[i, :n_pressure_iterations])
        averaged_physiological_effective_conductances[i] = physiological_effective_conductance
        if i == 0:
            averaged_stochastic_effective_conductances = np.zeros(stochastic_effective_conductances.shape[0])
            for j, (stochastic_effective_conductance, n_probability_iterations) in enumerate(zip(stochastic_effective_conductances, realized_n_probability_iterations)):
                if spontaneous_embolism:
                    averaged_stochastic_effective_conductances[j] = np.mean(stochastic_effective_conductance[i, :n_probability_iterations])
                else:
                    averaged_stochastic_effective_conductances[j] = np.mean(stochastic_effective_conductance[:n_probability_iterations])
        if not pool_physiological_only:
            optimized_spreading_probability_index = np.argmin(np.abs(averaged_stochastic_effective_conductances - physiological_effective_conductance)) 
            optimized_spreading_probabilities[i] = spreading_probability_range[optimized_spreading_probability_index]
            optimized_stochastic_effective_conductances[i] = averaged_stochastic_effective_conductances[optimized_spreading_probability_index]
            optimized_n_probability_iterations = realized_n_probability_iterations[optimized_spreading_probability_index]
        else:
            optimized_n_probability_iterations = 0
            optimized_spreading_probability_index = 0
        
        pressure_difference_phys_prevalences = [prevalences[i] for prevalences in physiological_prevalences]
        pressure_difference_phys_prevalences_due_to_spontaneous_embolism = [prevalences[i] for prevalences in physiological_prevalences_due_to_spontaneous_embolism]
        pressure_difference_phys_prevalences_due_to_spreading = [prevalences[i] for prevalences in physiological_prevalences_due_to_spreading]
        physiological_prevalence_length = max([len(prevalence) for prevalence in pressure_difference_phys_prevalences])
        for p, (prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading) in enumerate(zip(pressure_difference_phys_prevalences, pressure_difference_phys_prevalences_due_to_spontaneous_embolism, pressure_difference_phys_prevalences_due_to_spreading)):
            if len(prevalence) == 0:
                pressure_difference_phys_prevalences[p] = np.zeros(physiological_prevalence_length)
                pressure_difference_phys_prevalences_due_to_spontaneous_embolism[p] = np.zeros(physiological_prevalence_length)
                pressure_difference_phys_prevalences_due_to_spreading[p] = np.zeros(physiological_prevalence_length)
            elif len(prevalence) < physiological_prevalence_length: 
                pressure_difference_phys_prevalences[p] = np.concatenate((prevalence, prevalence[-1] * np.ones(physiological_prevalence_length - len(prevalence))))
                pressure_difference_phys_prevalences_due_to_spontaneous_embolism[p] = np.concatenate((prevalence_due_to_spontaneous_embolism, prevalence_due_to_spontaneous_embolism[-1] * np.ones(physiological_prevalence_length - len(prevalence_due_to_spontaneous_embolism))))
                pressure_difference_phys_prevalences_due_to_spreading[p] = np.concatenate((prevalence_due_to_spreading, prevalence_due_to_spreading[-1] * np.ones(physiological_prevalence_length - len(prevalence_due_to_spreading))))
        pressure_difference_phys_prevalences = np.array(pressure_difference_phys_prevalences)
        average_phys_prevalences.append(np.mean(pressure_difference_phys_prevalences[:n_pressure_iterations], axis=0))
        std_phys_prevalences.append(np.std(pressure_difference_phys_prevalences[:n_pressure_iterations], axis=0))
        pressure_difference_phys_prevalences_due_to_spontaneous_embolism = np.array(pressure_difference_phys_prevalences_due_to_spontaneous_embolism)
        average_phys_prevalences_due_to_spontaneous_embolism.append(np.mean(pressure_difference_phys_prevalences_due_to_spontaneous_embolism[:n_pressure_iterations], axis=0))
        std_phys_prevalences_due_to_spontaneous_embolism.append(np.std(pressure_difference_phys_prevalences_due_to_spontaneous_embolism[:n_pressure_iterations], axis=0))
        pressure_difference_phys_prevalences_due_to_spreading = np.array(pressure_difference_phys_prevalences_due_to_spreading)
        average_phys_prevalences_due_to_spreading.append(np.mean(pressure_difference_phys_prevalences_due_to_spreading[:n_pressure_iterations], axis=0))
        std_phys_prevalences_due_to_spreading.append(np.std(pressure_difference_phys_prevalences_due_to_spreading[:n_pressure_iterations], axis=0))
        
        for phys_prop, phys_av, phys_std in zip(phys_props, phys_avs, phys_stds):
            pressure_diff_phys_props = [props[i] for props in phys_prop]
            for p, props in enumerate(pressure_diff_phys_props):
                if len(props) == 0:
                    pressure_diff_phys_props[p] = np.zeros(physiological_prevalence_length)
                elif len(props) < physiological_prevalence_length:
                    pressure_diff_phys_props[p] = np.concatenate((props, props[-1] * np.ones(physiological_prevalence_length - len(props))))
            pressure_diff_phys_props = np.array(pressure_diff_phys_props)
            phys_av.append(np.mean(pressure_diff_phys_props[:n_pressure_iterations], axis=0))
            phys_std.append(np.std(pressure_diff_phys_props[:n_pressure_iterations], axis=0))
    
        if not pool_physiological_only:    
            if spontaneous_embolism:
                pressure_difference_stoch_prevalences = [prevalences[optimized_spreading_probability_index][i] for prevalences in stochastic_prevalences]
                pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = [prevalences[optimized_spreading_probability_index][i] for prevalences in stochastic_prevalences_due_to_spontaneous_embolism]
                pressure_difference_stoch_prevalences_due_to_spreading = [prevalences[optimized_spreading_probability_index][i] for prevalences in stochastic_prevalences_due_to_spreading]
            else:
                pressure_difference_stoch_prevalences = [prevalences[optimized_spreading_probability_index] for prevalences in stochastic_prevalences]
                pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = [prevalences[optimized_spreading_probability_index] for prevalences in stochastic_prevalences_due_to_spontaneous_embolism]
                pressure_difference_stoch_prevalences_due_to_spreading = [prevalences[optimized_spreading_probability_index] for prevalences in stochastic_prevalences_due_to_spreading]
            stochastic_prevalence_length = max([len(prevalence) for prevalence in pressure_difference_stoch_prevalences])
        else:
            pressure_difference_stoch_prevalences = []
            pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = []
            pressure_difference_stoch_prevalences_due_to_spreading = []
        for p, (prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading) in enumerate(zip(pressure_difference_stoch_prevalences, pressure_difference_stoch_prevalences_due_to_spontaneous_embolism, pressure_difference_stoch_prevalences_due_to_spreading)):
            if len(prevalence) == 0:
                pressure_difference_stoch_prevalences[p] = np.zeros(stochastic_prevalence_length)
                pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[p] = np.zeros(stochastic_prevalence_length)
                pressure_difference_stoch_prevalences_due_to_spreading[p] = np.zeros(stochastic_prevalence_length)
            elif len(prevalence) < stochastic_prevalence_length:
                pressure_difference_stoch_prevalences[p] = np.concatenate((prevalence, prevalence[-1] * np.ones(stochastic_prevalence_length - len(prevalence))))
                pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[p] = np.concatenate((prevalence_due_to_spontaneous_embolism, prevalence_due_to_spontaneous_embolism[-1] * np.ones(stochastic_prevalence_length - len(prevalence_due_to_spontaneous_embolism))))
                pressure_difference_stoch_prevalences_due_to_spreading[p] = np.concatenate((prevalence_due_to_spreading, prevalence_due_to_spreading[-1] * np.ones(stochastic_prevalence_length - len(prevalence_due_to_spreading))))
        pressure_difference_stoch_prevalences = np.array(pressure_difference_stoch_prevalences)
        average_stoch_prevalences.append(np.mean(pressure_difference_stoch_prevalences[:optimized_n_probability_iterations], axis=0))
        std_stoch_prevalences.append(np.std(pressure_difference_stoch_prevalences[:optimized_n_probability_iterations], axis=0))
        pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = np.array(pressure_difference_stoch_prevalences_due_to_spontaneous_embolism)
        average_stoch_prevalences_due_to_spontaneous_embolism.append(np.mean(pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[:optimized_n_probability_iterations], axis=0))
        std_stoch_prevalences_due_to_spontaneous_embolism.append(np.std(pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[:optimized_n_probability_iterations], axis=0))
        pressure_difference_stoch_prevalences_due_to_spreading = np.array(pressure_difference_stoch_prevalences_due_to_spreading)
        average_stoch_prevalences_due_to_spreading.append(np.mean(pressure_difference_stoch_prevalences_due_to_spreading[:optimized_n_probability_iterations], axis=0))
        std_stoch_prevalences_due_to_spreading.append(np.std(pressure_difference_stoch_prevalences_due_to_spreading[:optimized_n_probability_iterations], axis=0))

        for stoch_prop, stoch_av, stoch_std in zip(stoch_props, stoch_avs, stoch_stds):
            if not pool_physiological_only:
                if spontaneous_embolism:
                    pressure_diff_stoch_props = [props[optimized_spreading_probability_index][i] for props in stoch_prop]
                else:
                    pressure_diff_stoch_props = [props[optimized_spreading_probability_index] for props in stoch_prop]
            else:
                pressure_diff_stoch_props = []
            for p, props in enumerate(pressure_diff_stoch_props):
                if len(props) == 0:
                    pressure_diff_stoch_props[p] = np.zeros(stochastic_prevalence_length)
                elif len(props) < stochastic_prevalence_length:
                    pressure_diff_stoch_props[p] = np.concatenate((props, props[-1] * np.ones(stochastic_prevalence_length - len(props))))
            pressure_diff_stoch_props = np.array(pressure_diff_stoch_props)
            stoch_av.append(np.mean(pressure_diff_stoch_props[:optimized_n_probability_iterations], axis=0))
            stoch_std.append(np.std(pressure_diff_stoch_props[:optimized_n_probability_iterations], axis=0))

    data = {'pressure_differences': pressure_differences, 'optimized_spreading_probabilities': optimized_spreading_probabilities, 'physiological_effective_conductances': averaged_physiological_effective_conductances,
            'stochastic_effective_conductances': optimized_stochastic_effective_conductances, 'average_physiological_prevalences': average_phys_prevalences, 'std_physiological_prevalences': std_phys_prevalences,
            'average_stochastic_prevalences': average_stoch_prevalences, 'std_stochastic_prevalences': std_stoch_prevalences, 'average_physiological_prevalences_due_to_spontaneous_embolism': average_phys_prevalences_due_to_spontaneous_embolism,
            'std_physiological_prevalences_due_to_spontaneous_embolism': std_phys_prevalences_due_to_spontaneous_embolism, 'average_physiological_prevalences_due_to_spreading': average_phys_prevalences_due_to_spreading,
            'std_physiological_prevalences_due_to_spreading': std_phys_prevalences_due_to_spreading, 'average_stochastic_prevalences_due_to_spontaneous_embolism': average_stoch_prevalences_due_to_spontaneous_embolism,
            'std_stochastic_prevalences_due_to_spontaneous_embolism': std_stoch_prevalences_due_to_spontaneous_embolism, 'average_stochastic_prevalences_due_to_spreading': average_stoch_prevalences_due_to_spreading,
            'std_stochastic_prevalences_due_to_spreading': std_stoch_prevalences_due_to_spreading, 'average_physiological_lcc_size': av_phys_lcc_size, 'std_physiological_lcc_size': std_phys_lcc_size,
            'average_physiological_functional_lcc_size': av_phys_functional_lcc_size, 'std_physiological_functional_lcc_size': std_phys_functional_lcc_size, 'average_physiological_nonfunctional_component_size': av_phys_nonfunctional_component_size,
            'std_physiological_nonfunctional_component_size': std_phys_nonfunctional_component_size, 'average_physiological_nonfunctional_component_volume': av_phys_nonfunctional_component_volume, 'std_physiological_nonfunctional_component_volume': std_phys_nonfunctional_component_volume,
            'average_physiological_susceptibility': av_phys_susceptibility, 'std_physiological_susceptibility': std_phys_susceptibility, 'average_physiological_n_inlets': av_phys_n_inlets, 'std_physiological_n_inlets': std_phys_n_inlets,
            'average_physiological_n_outlets': av_phys_n_outlets, 'std_physiological_n_outlets': std_phys_n_outlets, 'average_stochastic_lcc_size': av_stoch_lcc_size, 'std_stochastic_lcc_size': std_stoch_lcc_size,
            'average_stochastic_functional_lcc_size': av_stoch_functional_lcc_size, 'std_stochastic_functional_lcc_size': std_stoch_functional_lcc_size, 'average_stochastic_nonfunctional_component_size': av_stoch_nonfunctional_component_size,
            'std_stochastic_nonfunctional_component_size': std_stoch_nonfunctional_component_size, 'average_stochastic_nonfunctional_component_volume': av_stoch_nonfunctional_component_volume, 'std_stochastic_nonfunctional_component_volume': std_stoch_nonfunctional_component_volume,
            'average_stochastic_susceptibility': av_stoch_susceptibility, 'std_stochastical_susceptibility': std_stoch_susceptibility, 'average_stochastic_functional_susceptibility': av_stoch_functional_susceptibility, 'std_stochastic_functional_susceptibility': std_stoch_functional_susceptibility,
            'average_stochastic_n_inlets': av_stoch_n_inlets, 'std_stochastic_n_inlets': std_stoch_n_inlets, 'average_stochastic_n_outlets': av_stoch_n_outlets, 'std_stochastic_n_outlets': std_stoch_n_outlets,
            'average_physiological_full_effective_conductances': average_phys_full_effective_conductances, 'std_physiological_full_effective_conductances': std_phys_full_effective_conductances,
            'average_stochastic_full_effective_conductances': average_stoch_full_effective_conductances, 'std_stochastic_full_effective_conductances': std_stoch_full_effective_conductances}
    
    with open(pooled_data_save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
    
def optimize_spreading_probability_against_empirical_vc(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, empirical_physiological_data_path, max_n_iterations=None):
    """
    Starting from an empirical vulnerability curve and previously simulated SI spreading, finds the optimal SI spreading probability corresponding to each pressure difference.

    Parameters
    ----------
    simulation_data_save_folder : str
        path of the folder in which the simulation data is saved
    simulation_data_save_name_base : str
        stem of the simulation data file names; the file names can contain other parts but files with names that don't contain this stem aren't read
    pooled_data_save_path : str
        path to which to save the pressure difference, optimal spreading probability and the effective conductance values corresponding to these should be saved
    empirical_physiological_data_path : str, optional
        path of the empirical physiological data
    max_n_iterations : int, optional
        the maximum number of iterations read per pressure difference or spreading probability (default None, in which case all iterations available are read)

    Returns
    -------
    None.
    """
    pressure_differences, spreading_probability_range, _, _, _, _, _, _, _, _, _, _, _, _, _, \
    stochastic_effective_conductances, stochastic_full_effective_conductances, stochastic_prevalences, \
    stochastic_prevalences_due_to_spontaneous_embolism, stochastic_prevalences_due_to_spreading, stochastic_lcc_size, stochastic_functional_lcc_size, stochastic_nonfunctional_component_size, stochastic_nonfunctional_component_volume, \
    stochastic_susceptibility, stochastic_functional_susceptibility, stochastic_n_inlets, stochastic_n_outlets, spontaneous_embolism, _, \
    realized_n_probability_iterations, physiological_PLCs = read_and_combine_spreading_probability_optimization_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, max_n_iterations=max_n_iterations, empirical_physiological_data=True, empirical_physiological_data_path=empirical_physiological_data_path)
    
    optimized_spreading_probabilities = np.zeros(len(pressure_differences))
    optimized_stochastic_PLCs = np.zeros(len(pressure_differences))

    average_stoch_full_effective_conductances = []
    std_stoch_full_effective_conductances = []
    average_stoch_prevalences = []
    std_stoch_prevalences = []
    average_stoch_prevalences_due_to_spontaneous_embolism = []
    std_stoch_prevalences_due_to_spontaneous_embolism = []
    average_stoch_prevalences_due_to_spreading = []
    std_stoch_prevalences_due_to_spreading = []
    av_stoch_lcc_size = []
    std_stoch_lcc_size = []
    av_stoch_functional_lcc_size = []
    std_stoch_functional_lcc_size = []
    av_stoch_nonfunctional_component_size = []
    std_stoch_nonfunctional_component_size = []
    av_stoch_nonfunctional_component_volume = []
    std_stoch_nonfunctional_component_volume = []
    av_stoch_susceptibility = []
    std_stoch_susceptibility = []
    av_stoch_functional_susceptibility = []
    std_stoch_functional_susceptibility = []
    av_stoch_n_inlets = []
    std_stoch_n_inlets = []
    av_stoch_n_outlets = []
    std_stoch_n_outlets = []
    
    stoch_props = [stochastic_full_effective_conductances, stochastic_lcc_size, stochastic_functional_lcc_size, stochastic_nonfunctional_component_size, stochastic_nonfunctional_component_volume, stochastic_susceptibility, stochastic_functional_susceptibility, stochastic_n_inlets, stochastic_n_outlets]
    stoch_avs = [average_stoch_full_effective_conductances, av_stoch_lcc_size, av_stoch_functional_lcc_size, av_stoch_nonfunctional_component_size, av_stoch_nonfunctional_component_volume, av_stoch_susceptibility, av_stoch_functional_susceptibility, av_stoch_n_inlets, av_stoch_n_outlets]
    stoch_stds = [std_stoch_full_effective_conductances, std_stoch_lcc_size, std_stoch_functional_lcc_size, std_stoch_nonfunctional_component_size, std_stoch_nonfunctional_component_volume, std_stoch_susceptibility, std_stoch_functional_susceptibility, std_stoch_n_inlets, std_stoch_n_outlets]
    
    for i, (pressure_difference, physiological_PLC) in enumerate(zip(pressure_differences, physiological_PLCs)):
        if i == 0:
            averaged_stochastic_effective_conductances = np.zeros(stochastic_effective_conductances.shape[0])
        for j, (stochastic_effective_conductance, n_probability_iterations) in enumerate(zip(stochastic_effective_conductances, realized_n_probability_iterations)):
            if spontaneous_embolism:
                averaged_stochastic_effective_conductances[j] = np.mean(stochastic_effective_conductance[i, :n_probability_iterations])
            else:
                averaged_stochastic_effective_conductances[j] = np.mean(stochastic_effective_conductance[:n_probability_iterations])
            stochastic_PLCs = (np.amax(averaged_stochastic_effective_conductances) - averaged_stochastic_effective_conductances) / np.amax(averaged_stochastic_effective_conductances)        
            if np.amax(physiological_PLCs) > 1:
                stochastic_PLCs = 100 * stochastic_PLCs # if physiological PLC is given as percentage (between 0 and 100), stochastic PLC should be percentage as well
                
        optimized_spreading_probability_index = np.argmin(np.abs(stochastic_PLCs - physiological_PLC))
        optimized_spreading_probabilities[i] = spreading_probability_range[optimized_spreading_probability_index]
        optimized_stochastic_PLCs[i] = stochastic_PLCs[optimized_spreading_probability_index]
        optimized_n_probability_iterations = realized_n_probability_iterations[optimized_spreading_probability_index]

        if spontaneous_embolism:
            pressure_difference_stoch_prevalences = [prevalences[optimized_spreading_probability_index][i] for prevalences in stochastic_prevalences]
            pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = [prevalences[optimized_spreading_probability_index][i] for prevalences in stochastic_prevalences_due_to_spontaneous_embolism]
            pressure_difference_stoch_prevalences_due_to_spreading = [prevalences[optimized_spreading_probability_index][i] for prevalences in stochastic_prevalences_due_to_spreading]
        else:
            pressure_difference_stoch_prevalences = [prevalences[optimized_spreading_probability_index] for prevalences in stochastic_prevalences]
            pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = [prevalences[optimized_spreading_probability_index] for prevalences in stochastic_prevalences_due_to_spontaneous_embolism]
            pressure_difference_stoch_prevalences_due_to_spreading = [prevalences[optimized_spreading_probability_index] for prevalences in stochastic_prevalences_due_to_spreading]
        stochastic_prevalence_length = max([len(prevalence) for prevalence in pressure_difference_stoch_prevalences])
        for p, (prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading) in enumerate(zip(pressure_difference_stoch_prevalences, pressure_difference_stoch_prevalences_due_to_spontaneous_embolism, pressure_difference_stoch_prevalences_due_to_spreading)):
            if len(prevalence) == 0:
                pressure_difference_stoch_prevalences[p] = np.zeros(stochastic_prevalence_length)
                pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[p] = np.zeros(stochastic_prevalence_length)
                pressure_difference_stoch_prevalences_due_to_spreading[p] = np.zeros(stochastic_prevalence_length)
            elif len(prevalence) < stochastic_prevalence_length:
                pressure_difference_stoch_prevalences[p] = np.concatenate((prevalence, prevalence[-1] * np.ones(stochastic_prevalence_length - len(prevalence))))
                pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[p] = np.concatenate((prevalence_due_to_spontaneous_embolism, prevalence_due_to_spontaneous_embolism[-1] * np.ones(stochastic_prevalence_length - len(prevalence_due_to_spontaneous_embolism))))
                pressure_difference_stoch_prevalences_due_to_spreading[p] = np.concatenate((prevalence_due_to_spreading, prevalence_due_to_spreading[-1] * np.ones(stochastic_prevalence_length - len(prevalence_due_to_spreading))))
        pressure_difference_stoch_prevalences = np.array(pressure_difference_stoch_prevalences)
        average_stoch_prevalences.append(np.mean(pressure_difference_stoch_prevalences[:optimized_n_probability_iterations], axis=0))
        std_stoch_prevalences.append(np.std(pressure_difference_stoch_prevalences[:optimized_n_probability_iterations], axis=0))
        pressure_difference_stoch_prevalences_due_to_spontaneous_embolism = np.array(pressure_difference_stoch_prevalences_due_to_spontaneous_embolism)
        average_stoch_prevalences_due_to_spontaneous_embolism.append(np.mean(pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[:optimized_n_probability_iterations], axis=0))
        std_stoch_prevalences_due_to_spontaneous_embolism.append(np.std(pressure_difference_stoch_prevalences_due_to_spontaneous_embolism[:optimized_n_probability_iterations], axis=0))
        pressure_difference_stoch_prevalences_due_to_spreading = np.array(pressure_difference_stoch_prevalences_due_to_spreading)
        average_stoch_prevalences_due_to_spreading.append(np.mean(pressure_difference_stoch_prevalences_due_to_spreading[:optimized_n_probability_iterations], axis=0))
        std_stoch_prevalences_due_to_spreading.append(np.std(pressure_difference_stoch_prevalences_due_to_spreading[:optimized_n_probability_iterations], axis=0))
        
        for stoch_prop, stoch_av, stoch_std in zip(stoch_props, stoch_avs, stoch_stds):
            if spontaneous_embolism:
                pressure_diff_stoch_props = [props[optimized_spreading_probability_index][i] for props in stoch_prop]
            else:
                pressure_diff_stoch_props = [props[optimized_spreading_probability_index] for props in stoch_prop]
            for p, props in enumerate(pressure_diff_stoch_props):
                if len(props) == 0:
                    pressure_diff_stoch_props[p] = np.zeros(stochastic_prevalence_length)
                elif len(props) < stochastic_prevalence_length:
                    pressure_diff_stoch_props[p] = np.concatenate((props, props[-1] * np.ones(stochastic_prevalence_length - len(props))))
            pressure_diff_stoch_props = np.array(pressure_diff_stoch_props)
            stoch_av.append(np.mean(pressure_diff_stoch_props[:optimized_n_probability_iterations], axis=0))
            stoch_std.append(np.std(pressure_diff_stoch_props[:optimized_n_probability_iterations], axis=0))

    data = {'pressure_differences': pressure_differences, 'optimized_spreading_probabilities': optimized_spreading_probabilities, 'physiological_PLCs': physiological_PLCs,
            'stochastic_PLCs': optimized_stochastic_PLCs, 'average_stochastic_prevalences': average_stoch_prevalences, 'std_stochastic_prevalences': std_stoch_prevalences, 
            'average_stochastic_prevalences_due_to_spontaneous_embolism': average_stoch_prevalences_due_to_spontaneous_embolism, 'std_stochastic_prevalences_due_to_spontaneous_embolism': std_stoch_prevalences_due_to_spontaneous_embolism, 
            'average_stochastic_prevalences_due_to_spreading': average_stoch_prevalences_due_to_spreading, 'std_stochastic_prevalences_due_to_spreading': std_stoch_prevalences_due_to_spreading,
            'average_stochastic_lcc_size': av_stoch_lcc_size, 'std_stochastic_lcc_size': std_stoch_lcc_size, 'average_stochastic_functional_lcc_size': av_stoch_functional_lcc_size, 'std_stochastic_functional_lcc_size': std_stoch_functional_lcc_size, 
            'average_stochastic_nonfunctional_component_size': av_stoch_nonfunctional_component_size, 'std_stochastic_nonfunctional_component_size': std_stoch_nonfunctional_component_size, 'average_stochastic_nonfunctional_component_volume': av_stoch_nonfunctional_component_volume, 'std_stochastic_nonfunctional_component_volume': std_stoch_nonfunctional_component_volume,
            'average_stochastic_susceptibility': av_stoch_susceptibility, 'std_stochastical_susceptibility': std_stoch_susceptibility, 'average_stochastic_functional_susceptibility': av_stoch_functional_susceptibility, 
            'std_stochastic_functional_susceptibility': std_stoch_functional_susceptibility, 'average_stochastic_n_inlets': av_stoch_n_inlets, 'std_stochastic_n_inlets': std_stoch_n_inlets, 
            'average_stochastic_n_outlets': av_stoch_n_outlets, 'std_stochastic_n_outlets': std_stoch_n_outlets, 'average_stochastic_full_effective_conductances': average_stoch_full_effective_conductances, 
            'std_stochastic_full_effective_conductances': std_stoch_full_effective_conductances}

    with open(pooled_data_save_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()
    
    
def read_and_combine_spreading_probability_optimization_data(simulation_data_save_folder, simulation_data_save_name_base, pooled_data_save_path, max_n_iterations=None, empirical_physiological_data=False, empirical_physiological_data_path=''):
    """
    Reads spreading probability data files that may contain different pressure and probability ranges and combines their data for optimizing
    spreading probability.
    
    Parameters
    ----------
    simulation_data_save_folder : str
        path of the folder in which the simulation data is saved
    simulation_data_save_name_base : str
        stem of the simulation data file names; the file names can contain other parts but files with names that don't contain this stem aren't read
    pooled_data_save_path : str
        path to which to save the pressure difference, optimal spreading probability and the effective conductance values corresponding to these should be saved
        (used here to omit possible pooled data pickles from reading)
    max_n_iterations : int, optional
        the largest number of iterations that is read for each pressure difference / spreading probability (default None, in which case all available iterations are read)
    empirical_physiological_data : bln, optional
        option for reading empirical physiological conductances instead of simulated ones. If this is set to True, physiological vulnerability curve is read from
        empirical_physiological_data_path. Note that in this case, only the final percentage of effective conductance lost at each pressure diff (the vulnerability curve) is
        read, while empty lists are returned for physiological_full_effective_conductances and network properties
    empirical_physiological_data_path : str, optional
        path of the empirical physiological data (default: '')

    Returns
    -------
    pressure_differences : np.array
        all pressure differences saved in the simulation data files
    spreading_probability_range : np.array
        all spreading probabilities saved in the simulation data files
    physiological_effective_conductances : np.array
        final effective conductances, shape n_pressure_differences x n_iterations
        NOTE that if empirical_physiological_data == True, this contains the percentage of conductance lost (PLC) instead of the absolute effective conductance
    physiological_full_effective_conductances : list of lists
        evolution of effective conductance for each pressure difference, n_iterations x n_pressures x time
    physiological_prevalences : list of lists
        evolution of prevalence for each pressure difference, n_iterations x n_pressures x time
    physiological_prevalences_due_to_spontaneous_embolism : list of lists
        evolution of prevalence due to spontaneous embolism for each pressure difference, n_iterations x n_pressures x time
    physiological_prevalences_due_to_spreading : list of lists
        evolution of prevalence due to spreading for each pressure difference, n_iterations x n_pressures x time
    physiological_lcc_size : list of lists
        evolution of LCC size for each pressure difference, n_iterations x n_pressures x time
    physiological_functional_lcc_size : list of lists
        evolution of functional LCC size for each pressure difference, n_iterations x n_pressures x time
    physiological_nonfunctional_component_size : list of lists
        evolution of nonfunctional component size for each pressure difference, n_iterations x n_pressures x time
    physiological_nonfunctional_component_volume : list of lists
        evolution of nonfunctional component volume for each pressure difference, n_iterations x n_pressures x time
    physiological_susceptibility : list of lists
        evolution of susceptibility for each pressure difference, n_iterations x n_pressures x time
    physiological_functional_susceptibility : list of lists
        evolution of functional susceptibility for each pressure difference, n_iterations x n_pressures x time
    physiological_n_inlets : list of lists
        evolution of n_inlets for each pressure difference, n_iterations x n_pressures x time
    physiological_n_outlets : list of lists
        evolution of n_outlets for each pressure difference, n_iterations x n_pressures x time
    stochastic_effective_conductances : np.array
        final effective conductances, shape n_probabilities x n_iterations (if there is no spontaneous embolism) or n_probabilities x n_pressures x n_iterations (if spontaneous embilism is present)''
    stochastic_full_effective_conductances : list of lists
        evolution of effective conductance for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_prevalences : list of lists
        evolution of prevalence for each spreading probability, n_iterations x n_pressures x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_prevalences_due_to_spontaneous_embolism : list of lists
        evolution of prevalence  due to spontaneous embolism for each spreading probability, n_iterations x n_pressures x time (in which case there is no spontaneous embolism => all values are 0) or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_prevalences_due_to_spreading : list of lists
        evolution of prevalence due to spreading for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_lcc_size : list of lists
        evolution of LCC size for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_functional_lcc_size : list of lists
        evolution of functional LCC size for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_nonfunctional_component_size : list of lists
        evolution of nonfunctional component size for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_nonfunctional_component_volume : list of lists
        evolution of nonfunctional component volume for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_susceptibility : list of lists
        evolution of suscpetibility for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_functional_susceptibility : list of lists
        evolution of functional susceptibility for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_n_inlets : list of lists
        evolution of n_inlets for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    stochastic_n_outlets : list of lists
        evolution of n_outlets for each spreading_probability, n_iterations x n_probabilities x time or n_itertions x n_iterations x n_pressures x n_probabilities x time
    spontaneous_embolism : bln
        are there spontaneous embolisms present in the data
    realized_n_pressure_iterations : np.array of ints
        for each pressure difference, the number of iterations read
    realized_n_probability_iterations : np.array of ints
        for each spreading probability, the numver of iterations read
    physiological_PLC : np.array  of floats
        for each pressure differences, the empirical percentage of lost conductance. Note that PLC values are returned only if empirical_physiological_data == True, otherwise
        [] is returned instead
    """
    raw_phys_eff_conductances = []
    raw_phys_full_eff_conductances = []
    raw_phys_prevalence = []
    raw_phys_prevalence_due_to_spontaneous_embolism = []
    raw_phys_prevalence_due_to_spreading = []
    raw_phys_lcc_size = []
    raw_phys_func_lcc_size = []
    raw_phys_nonfunc_component_size = []
    raw_phys_nonfunc_component_volume = []
    raw_phys_susceptibility = []
    raw_phys_func_susceptibility = []
    raw_phys_n_inlets = []
    raw_phys_n_outlets = []
    raw_stoch_eff_conductances = []
    raw_stoch_full_eff_conductances = []
    raw_stoch_prevalence = []
    raw_stoch_prevalence_due_to_spontaneous_embolism = []
    raw_stoch_prevalence_due_to_spreading = []
    raw_stoch_lcc_size = []
    raw_stoch_func_lcc_size = []
    raw_stoch_nonfunc_component_size = []
    raw_stoch_nonfunc_component_volume = []
    raw_stoch_susceptibility = []
    raw_stoch_func_susceptibility = []
    raw_stoch_n_inlets = []
    raw_stoch_n_outlets = []

    physiological_PLC = []
    
    data_pressure_differences = []
    data_spreading_probabilities = []
    data_spontaneous_embolism_pressure_differences = []
    
    phys_properties = [raw_phys_eff_conductances, raw_phys_full_eff_conductances, raw_phys_func_lcc_size, raw_phys_func_susceptibility, raw_phys_lcc_size, raw_phys_n_inlets, raw_phys_n_outlets, raw_phys_nonfunc_component_size, raw_phys_nonfunc_component_volume,
                  raw_phys_prevalence, raw_phys_prevalence_due_to_spontaneous_embolism, raw_phys_prevalence_due_to_spreading, raw_phys_susceptibility]
    stoch_properties = [raw_stoch_eff_conductances, raw_stoch_full_eff_conductances, raw_stoch_func_lcc_size,
                  raw_stoch_func_susceptibility, raw_stoch_lcc_size, raw_stoch_n_inlets, raw_stoch_n_outlets, raw_stoch_nonfunc_component_size, raw_stoch_nonfunc_component_volume, raw_stoch_prevalence, raw_stoch_prevalence_due_to_spontaneous_embolism,
                  raw_stoch_prevalence_due_to_spreading, raw_stoch_susceptibility]
    phys_keys = ['physiological_effective_conductances', 'physiological_full_effective_conductances', 'physiological_functional_lcc_size', 'physiological_functional_susceptibility', 'physiological_lcc_size', 'physiological_n_inlets',
            'physiological_n_outlets', 'physiological_nonfunctional_component_size', 'physiological_nonfunctional_component_volume', 'physiological_prevalences', 'physiological_prevalences_due_to_spontaneous_embolism',
            'physiological_prevalences_due_to_spreading', 'physiological_susceptibility']
    stoch_keys = ['stochastic_effective_conductances', 'stochastic_full_effective_conductances', 'stochastic_functional_lcc_size', 'stochastic_functional_susceptibility',
            'stochastic_lcc_size', 'stochastic_n_inlets', 'stochastic_n_outlets', 'stochastic_nonfunctional_component_size', 'stochastic_nonfunctional_component_volume', 'stochastic_prevalences', 'stochastic_prevalences_due_to_spontaneous_embolism',
            'stochastic_prevalences_due_to_spreading', 'stochastic_susceptibility']
    
    if empirical_physiological_data:
        with open(empirical_physiological_data_path, 'rb') as f:
            empirical_data = pickle.load(f)
            f.close()
        data_pressure_differences = empirical_data['pressure_differences']
        physiological_PLC.extend(empirical_data['PLC'])
        physiological_PLC = np.array(physiological_PLC)
        
    data_files = [os.path.join(simulation_data_save_folder, file) for file in os.listdir(simulation_data_save_folder) if os.path.isfile(os.path.join(simulation_data_save_folder, file))]
    data_files = [data_file for data_file in data_files if simulation_data_save_name_base in data_file]
    if pooled_data_save_path in data_files:
        data_files.remove(pooled_data_save_path)
    for i, data_file in enumerate(data_files):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            f.close()
        if i == 0:
            spontaneous_embolism = data['spontaneous_embolism']
        else:
            assert data['spontaneous_embolism'] == spontaneous_embolism, 'Please do not mix spreading probability optimization data with and without spontaneous embolism'
        if not empirical_physiological_data:
            if len(data['pressure_differences']) > 0:
                for phys_prop, phys_key in zip(phys_properties, phys_keys):
                    phys_prop.extend(data[phys_key])
                data_pressure_differences.extend(data['pressure_differences'])
            
        if len(data['spreading_probability_range']) > 0:
            for stoch_prop, stoch_key in zip(stoch_properties, stoch_keys):
                stoch_prop.extend(data[stoch_key])
            data_spreading_probabilities.extend(data['spreading_probability_range'])
            if spontaneous_embolism:
                data_spontaneous_embolism_pressure_differences.extend(data['spontaneous_embolism_pressure_differences'])
    
    pressure_differences, realized_n_pressure_iterations = np.unique(np.round(data_pressure_differences, decimals=10), return_counts=True) # rounding to avoid float accuracy issues
    spreading_probability_range, realized_n_probability_iterations = np.unique(np.round(data_spreading_probabilities, decimals=10), return_counts=True)
    if spontaneous_embolism:
        spontaneous_embolism_pressure_differences = np.unique(np.round(data_spontaneous_embolism_pressure_differences, decimals=10))
        assert np.all(spontaneous_embolism_pressure_differences == pressure_differences), 'pressure differences used for calculating spontaneous embolism probabilities do not match the pressure differences used to simulate spreading, please check the data'
    
    if max_n_iterations == None:
        n_iterations = max(np.amax(realized_n_pressure_iterations), np.amax(realized_n_probability_iterations))
    else:
        n_iterations = max_n_iterations
        realized_n_pressure_iterations[np.where(realized_n_pressure_iterations > max_n_iterations)] = max_n_iterations
        realized_n_probability_iterations[np.where(realized_n_probability_iterations > max_n_iterations)] = max_n_iterations
            
    pressure_differences = np.sort(pressure_differences)
    spreading_probability_range = np.sort(spreading_probability_range)
    
    physiological_effective_conductances = np.zeros((len(pressure_differences), n_iterations))
    physiological_full_effective_conductances = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_prevalences = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_prevalences_due_to_spontaneous_embolism = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_prevalences_due_to_spreading = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_lcc_size = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_functional_lcc_size = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_nonfunctional_component_size = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_nonfunctional_component_volume = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_susceptibility = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_functional_susceptibility = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_n_inlets = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    physiological_n_outlets = [[[] for pressure_diff in pressure_differences] for i in range(n_iterations)]
    if spontaneous_embolism:
        stochastic_effective_conductances = np.zeros((len(spreading_probability_range), len(pressure_differences), n_iterations)) 
        stochastic_full_effective_conductances = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_prevalences = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_prevalences_due_to_spontaneous_embolism = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_prevalences_due_to_spreading = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_lcc_size = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_functional_lcc_size = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_nonfunctional_component_size = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_nonfunctional_component_volume = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_susceptibility = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_functional_susceptibility = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_n_inlets = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_n_outlets = [[[[] for pressure_difference in pressure_differences] for probability in spreading_probability_range] for i in range(n_iterations)]
    else:
        stochastic_effective_conductances = np.zeros((len(spreading_probability_range), n_iterations))
        stochastic_full_effective_conductances = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_prevalences = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_prevalences_due_to_spontaneous_embolism = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_prevalences_due_to_spreading = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_lcc_size = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_functional_lcc_size = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_nonfunctional_component_size = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_nonfunctional_component_volume = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_susceptibility = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_functional_susceptibility = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_n_inlets = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
        stochastic_n_outlets = [[[] for probability in spreading_probability_range] for i in range(n_iterations)]
    
    out_phys_properties = [physiological_full_effective_conductances, physiological_functional_lcc_size, physiological_functional_susceptibility, physiological_lcc_size, physiological_n_inlets, physiological_n_outlets, 
                           physiological_nonfunctional_component_size, physiological_nonfunctional_component_volume, physiological_prevalences, physiological_prevalences_due_to_spontaneous_embolism, physiological_prevalences_due_to_spreading, 
                           physiological_susceptibility]
    out_stoch_properties = [stochastic_full_effective_conductances, stochastic_functional_lcc_size, stochastic_functional_susceptibility, stochastic_lcc_size, stochastic_n_inlets, stochastic_n_outlets, 
                            stochastic_nonfunctional_component_size, stochastic_nonfunctional_component_volume, stochastic_prevalences, stochastic_prevalences_due_to_spontaneous_embolism, stochastic_prevalences_due_to_spreading, 
                            stochastic_susceptibility]
    phys_properties.remove(raw_phys_eff_conductances)
    stoch_properties.remove(raw_stoch_eff_conductances)
    
    if not empirical_physiological_data:
        phys_iteration = np.zeros(len(pressure_differences), dtype=int)
    
        for i, data_pressure_diff in enumerate(data_pressure_differences):
            data_pressure_diff = np.round(data_pressure_diff, decimals=10) # rounding because of float accuracy issues
            index = np.where(pressure_differences == data_pressure_diff)[0][0]
            if (not max_n_iterations == None) and (phys_iteration[index] >= max_n_iterations):
                continue
            physiological_effective_conductances[index, phys_iteration[index]] = raw_phys_eff_conductances[i]
            for phys_prop, out_phys_prop in zip(phys_properties, out_phys_properties):
                out_phys_prop[phys_iteration[index]][index] = phys_prop[i]
            phys_iteration[index] += 1
        
    stoch_iteration = np.zeros(len(spreading_probability_range), dtype=int)
    
    for i, data_probability in enumerate(data_spreading_probabilities):
        data_probability = np.round(data_probability, decimals=10)
        index = np.where(spreading_probability_range == data_probability)[0][0]
        if (not max_n_iterations == None) and (stoch_iteration[index] >= max_n_iterations):
            continue
        if spontaneous_embolism:
            for j, data_pressure_diff in enumerate(data_pressure_differences): 
                pressure_index = np.where(pressure_differences == data_pressure_diff)[0][0]
                stochastic_effective_conductances[index, pressure_index, stoch_iteration[index]] = raw_stoch_eff_conductances[i][j]
                for stoch_prop, out_stoch_prop in zip(stoch_properties, out_stoch_properties):
                    out_stoch_prop[stoch_iteration[index]][index][pressure_index] = stoch_prop[i][j]
        else:
            stochastic_effective_conductances[index, stoch_iteration[index]] = raw_stoch_eff_conductances[i]
            for stoch_prop, out_stoch_prop in zip(stoch_properties, out_stoch_properties):
                out_stoch_prop[stoch_iteration[index]][index] = stoch_prop[i]
        stoch_iteration[index] += 1
        
    return pressure_differences, spreading_probability_range, physiological_effective_conductances, physiological_full_effective_conductances, physiological_prevalences, physiological_prevalences_due_to_spontaneous_embolism, \
           physiological_prevalences_due_to_spreading, physiological_lcc_size, physiological_functional_lcc_size, physiological_nonfunctional_component_size, physiological_nonfunctional_component_volume, physiological_susceptibility, \
           physiological_functional_susceptibility, physiological_n_inlets, physiological_n_outlets, stochastic_effective_conductances, stochastic_full_effective_conductances, stochastic_prevalences, \
           stochastic_prevalences_due_to_spontaneous_embolism, stochastic_prevalences_due_to_spreading, stochastic_lcc_size, stochastic_functional_lcc_size, stochastic_nonfunctional_component_size, stochastic_nonfunctional_component_volume, \
           stochastic_susceptibility, stochastic_functional_susceptibility, stochastic_n_inlets, stochastic_n_outlets, spontaneous_embolism, realized_n_pressure_iterations, \
           realized_n_probability_iterations, physiological_PLC
    
def run_conduit_si_repeatedly(net, net_proj, cfg, spreading_param=0, include_orig_values=False):
    """
    Re-initializes all objects of op.Network().project to have given initial properties and
    calls run_conduit_si. This is a helper function used by optimize_spreading_probability.

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements, throats to connections between the elements
    net_proj : list of dics
        desired properties of each object belonging to net.project; properties of each object are given as a dictionary
    cfg : dict
        contains:
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
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
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6); only used if bpp_type == 'young-laplace'
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2); only used if bpp_type == 'young-laplace'
        average_pit_area : float, the average area of a pit
        nCPUs : int, number of CPUs used for parallel computing (default 5)
        bpp_type: str, how the bubble propagation pressure is calculated; options: 'young-laplace' (i.e. as in Mrad et al. 2018) and 'young-laplace_with_constrictions' (i.e. as in Kaack et al. 2021)
        bpp_data_path : str, optional, path, to which the BPP data has been saved; only used if bpp_type == 'young-laplace_with_constrictions'
    spreading_param : float
        parameter that controls the spreading speed, specifically
        if si_type == 'stochastic', spreading_param is the probability at which embolism spreads to neighbouring conduits (default: 0.1)
        if si_type == 'physiological', spreading param is difference between water pressure and vapour-air bubble pressure, delta P in the Mrad et al. article (default 0)
    include_orig_values : bln, optional
        should the output arrays include also values calculated for the intact network in addition to the values after each removal (default: False)

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
    prevalence : np.array
        fraction of embolized conduits at each infection step
    """   
    for proj_member in net.project[::-1]:
        if proj_member.name not in net_proj.keys():
            net.project.remove(proj_member)
            
    for proj_member, member_properties in net_proj.items():
        for prop in member_properties.keys():
            net.project[proj_member][prop] = member_properties[prop]

    effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_volume, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading = run_conduit_si(net, cfg, spreading_param, include_orig_values)
    return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_volume, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence
        
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
            conduit_elements = mrad_model.get_conduit_elements(net=sim_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            pore_diameter = sim_net['pore.diameter']
            sim_net, removed_components = mrad_model.clean_network(sim_net, np.concatenate((conduit_elements[:,0:3]/conduit_element_length, conduit_elements[:,3::]),axis=1), cfg['net_size'][0] - 1, remove_dead_ends=False)
            nonfunctional_component_size[i] = np.sum([len(removed_component) for removed_component in removed_components])
            removed_elements = list(itertools.chain.from_iterable(removed_components))
            nonfunctional_component_volume[i] = np.sum(np.pi * 0.5 * pore_diameter[removed_elements]**2 * conduit_element_length)
            n_inlets[i], n_outlets[i] = get_n_inlets(sim_net, (cfg['net_size'][0] - 1)*conduit_element_length, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            mrad_model.prepare_simulation_network(sim_net, cfg, update_coords=False)
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
            conduit_elements = mrad_model.get_conduit_elements(net=perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            perc_net, removed_components = mrad_model.clean_network(perc_net, np.concatenate((conduit_elements[:,0:3]/conduit_element_length, conduit_elements[:,3::]),axis=1), cfg['net_size'][0] - 1, remove_dead_ends=False)
            removed_elements = list(itertools.chain.from_iterable(removed_components))
            nonfunctional_component_volume[i] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * pore_diameter[removed_elements]**2 * conduit_element_length)
            if len(removed_components) > 0:
                removed_lcc = max([len(component) for component in removed_components])
                if removed_lcc > max_removed_lcc: # Percolation doesn't affect the sizes of removed components -> the largest removed component size changes only if a new, larger component gets removed
                    max_removed_lcc = removed_lcc
            if lcc_size[i] < max_removed_lcc:
                lcc_size[i] = max_removed_lcc
            nonfunctional_component_size[i] = nonfunctional_component_size[i - 1] + np.sum([len(removed_component) for removed_component in removed_components])
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, (cfg['net_size'][0] - 1)*conduit_element_length, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            mrad_model.prepare_simulation_network(perc_net, cfg, update_coords=False)
            effective_conductances[i], _ = simulations.simulate_water_flow(perc_net, cfg, visualize=False)
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
            lcc_size[i], susceptibility[i], _ = get_conduit_lcc_size(net=perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(net=perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns']) # orig_perc_net is needed for calculating measures related to non-functional components (these are removed from perc_net before the simulations)
            orig_perc_net['throat.type'] = perc_net['throat.type']
            orig_perc_net['pore.diameter'] = perc_net['pore.diameter']
            perc_net, removed_components = mrad_model.clean_network(perc_net, np.concatenate((conduit_elements[:,0:3]/conduit_element_length, conduit_elements[:,3::]),axis=1), cfg['net_size'][0] - 1, remove_dead_ends=False)
            removed_elements = list(itertools.chain.from_iterable(removed_components)) # calculating the size of the largest removed component in conduits
            nonfunctional_component_volume[i] = nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'][removed_elements]**2 * conduit_element_length)
            if len(removed_elements) > 0:
                removed_net = get_induced_subnet(orig_perc_net, removed_elements, return_network=True)
                removed_lcc, _, n_nonfunctional_conduits = get_conduit_lcc_size(net=removed_net, use_cylindrical_coords=use_cylindrical_coords, 
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
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, (cfg['net_size'][0] - 1)*conduit_element_length, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            mrad_model.prepare_simulation_network(perc_net, cfg, update_coords=False)
            effective_conductances[i], _ = simulations.simulate_water_flow(perc_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i], _ = get_conduit_lcc_size(net=perc_net)
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

def run_conduit_si(net, cfg, spreading_param=0, include_orig_values=False):
    """
    Starting from a given conduit, simulates an SI or SEI (embolism) spreading process on the conduit network. The spreading can be stochastic (at each step, each conduit is
    embolized at a certain probability that depends on if their neighbours have been embolized) or physiological (at each step, each conduit is
    embolized if it has embolized neighbours and the pressure difference with the neighbours is large enough). If the SI process is simulated,
    conduits become directly embolized with the given probability / at the given pressure difference. On the other hand, if the SEI process is simulated,
    conduits become first exposed and get embolized later with a probability given as a parameter.

    Parameters
    ----------
    net : op.Network
        pores correspond to conduit elements, throats to connections between them
    cfg : dict
        contains:
        net_size: np.array, size of the network to be created (n_rows x n_columns x n_depths)
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits
        conduit_diameters: np.array of floats, diameters of the conduits, or 'lognormal'
        to draw diameters from a lognormal distribution defin3ed by Dc and Dc_cv
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
        si_type : str 
            'stochastic' for probability-based spreading (default option)
            'physiological' for spreading based on pressure differences
            'stochastic_sei' for probability-based S-E transtion + E-I transition
            'physiological_sei' for pressure-based S-E transition + E-I transition
        weibull_a : float, Weibull distribution scale parameter (used for simulating pressure-difference-based embolism spreading) (default 20.28E6); only used if bpp_type == 'young-laplace'
        weibull_b : float, Weibull distribution shape parameter (used for simulating pressure-difference-based embolism spreading) (default 3.2); only used if bpp_type ==  'young-laplace'
        average_pit_area : float, the average area of a pit
        start_conduits : str or array-like of ints, the first conduits to be removed (i.e. the first infected node of the simulation)
            if 'random', a single start conduit is selected at random
            if 'random_per_component', a single start conduit per network component is selected at random
            if 'none', no start conduits are selected and embolism spreading is based solely on spontaneous embolism (for using this option, spontaneous_embolism must be True)
            if an array-like of ints is given, the ints are used as indices of the start conduits
        si_length : int, maximum number of time steps used for the simulation (default: 1000)
        spontaneous_embolism : bln, is spontaneous embolism through bubble expansion allowed (default: False)
        spontaneous_embolism_probability : float, probability of spontaneous embolism
        bubble_expansion_probability: float, probability of the E-I transition
        bpp_type: str, how the bubble propagation pressure is calculated; options: 'young-laplace' (i.e. as in Mrad et al. 2018) and 'young-laplace_with_constrictions' (i.e. as in Kaack et al. 2021)
        bpp_data_path : str, optional, path, to which the BPP data has been saved; only used if bpp_type == 'young-laplace_with_constrictions'
    spreading_param : float
        parameter that controls the spreading speed, specifically
        if si_type == 'stochastic', spreading_param is the probability at which embolism spreads to neighbouring conduits (default: 0.1)
        if si_type == 'physiological', spreading param is difference between water pressure and vapour-air bubble pressure, delta P in the Mrad et al. article (default 0)
    include_orig_values : bln, optional
        should the output arrays include also values calculated for the intact network in addition to the values after each removal (default: False)
    
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
    prevalence_due_to_spontaneous_embolism : np.array
        fraction of spontaneously embolized conduits at each infection step
    prevalence_due_to_spreading : np.array
        fraction of conduits embolized through spreading at each infection step (the start conduit contribute to this)
    fraction_of_exposed : np.array
        fraction of exposed conduits at each infection step (only returned if si_type is stochastic_sei or physiological_sei)

    """
    si_type = cfg.get('si_type', 'stochastic')
    assert si_type in ['stochastic', 'physiological', 'stochastic_sei', 'physiological_sei'], 'unknown si type, select stochastic, stochastic_sei, physiological or physiological_sei'
    assert len(net['pore.diameter']) > 0, 'pore diameters not defined; please define pore diameters before running percolation'
    bpp_type = cfg['bpp_type']
    assert bpp_type in ['young-laplace', 'young-laplace_with_constrictions'], 'unknown BPP type; please select young-laplace or young-laplace_with_constrictions'
    if bpp_type == 'young-laplace_with_constrictions':
        bpp_data_path = cfg['bpp_data_path']
    conduit_element_length = cfg.get('conduit_element_length', params.Lce)
    heartwood_d = cfg.get('heartwood_d', params.heartwood_d)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', False)
    si_tolerance_length = cfg.get('si_tolerance_length', 20)
    si_length = cfg.get('si_length', 1000)
    spontaneous_embolism = cfg.get('spontaneous_embolism', False)
    if spontaneous_embolism:
        spontaneous_embolism_probability = cfg['spontaneous_embolism_probability']
        assert 0 <= spontaneous_embolism_probability <= 1, 'spontaneous embolism probability must be between 0 and 1'
    if si_type in ['stochastic_sei', 'physiological_sei']:
        bubble_expansion_probability = cfg['bubble_expansion_probability']
        assert 0 <= bubble_expansion_probability <= 1, 'bubble expansion probability must be between 0 and 1'
        if bubble_expansion_probability == 0:
            import warnings
            warnings.warn('Bubble expansion probability is set to 0 in an SEI spreading process. The simulation will run but no conduits will become embolized.')
    conns = net['throat.conns']
    assert len(conns) > 0, 'Network has no throats; cannot run percolation analysis'
    conn_types = net['throat.type']
    cec_mask = conn_types == cec_indicator
    cec = conns[cec_mask]
    conduits = mrad_model.get_conduits(cec)
    orig_conduits = conduits.copy()
    
    if si_type in ['stochastic', 'stochastic_sei']:
        if spreading_param >= 0:
            spreading_probability = spreading_param
        else:
            spreading_probability = 0.1
    elif si_type in ['physiological', 'physiological_sei']:
        pressure_diff = spreading_param
        
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
    prevalence_due_to_spreading = np.zeros(si_length)
    prevalence_due_to_spontaneous_embolism = np.zeros(si_length)
    fraction_of_exposed = np.zeros(si_length)
    
    np.random.seed() # this is to ensure that different parallel calls produce different start conduits

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
        elif start_conduits == 'none':
            assert spontaneous_embolism, 'no start conduits given and spontaneous embolism not allowed so simulating embolism spreading is not possible'
            start_conduits = []
    
    embolization_times = np.zeros((conduits.shape[0], 3))
    embolization_times[:, 0] = np.inf
    embolization_times[:, 1] = 1 # the second column indicates if the conduit is functional 
    embolization_times[:, 2] = 1 # the third colum indicates the compartment of the conduit (1: susceptible, 0: exposed, -1: emoblized)

    for start_conduit in start_conduits:
        embolization_times[start_conduit, 0] = 0
        embolization_times[start_conduit, 1] = 0
    conduit_neighbours = get_conduit_neighbors(net, use_cylindrical_coords, conduit_element_length, heartwood_d, cec_indicator)
    
    perc_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
    perc_net['throat.type'] = net['throat.type']
    if 'pore.diameter' in net.keys():
        perc_net['pore.diameter'] = net['pore.diameter']

    orig_pore_diameter = np.copy(net['pore.diameter'])
    if si_type == 'physiological':
        if bpp_type == 'young-laplace':
            bpp = calculate_bpp(net, conduits, 1 - cec_mask, cfg)
        elif bpp_type == 'young-laplace_with_constrictions':
            bpp = pit_membrane.read_constriction_bpp(bpp_data_path, net, conduits, 1 - cec_mask, cfg)
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
    last_removed_by_embolism = False
    n_embolized_through_spreading = len(start_conduits)
    n_spontaneously_embolized = 0
    
    if include_orig_values:
        mrad_model.prepare_simulation_network(perc_net, cfg, update_coords=False)
        orig_effective_conductance, _ = simulations.simulate_water_flow(perc_net, cfg, visualize=False, return_water=False)
        orig_lcc_size, orig_susceptibility, _ = get_conduit_lcc_size(net=perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
        orig_functional_lcc_size = orig_lcc_size
        orig_functional_susceptibility = orig_susceptibility
        orig_nonfunctional_component_size = 0
        orig_n_inlets, orig_n_outlets = get_n_inlets(perc_net, (cfg['net_size'][0] - 1)*conduit_element_length, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
        orig_nonfunctional_component_volume = 0
        orig_prevalence = 0
        orig_prevalence_due_to_spontaneous_embolism = 0
        orig_prevalence_due_to_spreading = 0
        
        if si_type in ['stochastic_sei', 'physiological_sei']:
            orig_fraction_of_exposed = 0
         
    while (prevalence_diff > 0) & (time_step < si_length):

        pores_to_remove = []
        removed_conduit_indices = []

        if time_step == 0:
            for start_conduit in np.sort(start_conduits)[::-1]:
                pores_to_remove.extend(list(np.arange(conduits[start_conduit][0], conduits[start_conduit][1] + 1)))
                conduits[start_conduit +1::, 0:2] = conduits[start_conduit +1::, 0:2] - conduits[start_conduit, 2]
            conduits[start_conduits, :] = -1
        else: 
            # spontaneous embolism
            if spontaneous_embolism:
                test = np.random.rand(conduits.shape[0])
                embolization = test < spontaneous_embolism_probability
                spontaneously_embolized_conduits = np.where(embolization)[0]
                for spontaneously_embolized_conduit in spontaneously_embolized_conduits:
                    if embolization_times[spontaneously_embolized_conduit, 0] > time_step:
                        embolization_times[spontaneously_embolized_conduit, 0] = time_step
                        n_spontaneously_embolized += 1
                        if embolization_times[spontaneously_embolized_conduit, 1] > 0: # the spontaneously embolized conduit is functional and will be removed from the network
                            embolization_times[spontaneously_embolized_conduit, 1] = 0
                            pores_to_remove.extend(list(np.arange(conduits[spontaneously_embolized_conduit, 0], conduits[spontaneously_embolized_conduit, 1] + 1)))
                            removed_conduit_indices.append(spontaneously_embolized_conduit)
                        else: # if a nonfunctional conduit is embolized, nonfunctional component size and volume decrease
                            nonfunctional_component_size[time_step] -= 1
                            nonfunctional_component_volume[time_step] -= np.sum(np.pi * 0.5 * orig_pore_diameter[np.arange(orig_conduits[spontaneously_embolized_conduit, 0], orig_conduits[spontaneously_embolized_conduit, 1] + 1)]**2 * conduit_element_length)
            
            # embolism spreading
            embolized_conduits = np.where(embolization_times[:, 0] < time_step)[0]
            if not spontaneous_embolism and (np.sum(embolization_times[:, 2] == 0) == 0): # this stops the simulation if further embolisations are not possible: there are no unembolised conduits with embolised neighbours or exposed conduits and spontaneous embolism is not allowed
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
                if embolization_times[conduit, 0] <= time_step:
                    continue # conduit is already embolized
                else:
                    embolize = False
                    if si_type in ['stochastic_sei', 'physiological_sei']: # SEI spreading process
                        if embolization_times[conduit, 2] == 0: # conduit is exposed
                            embolize = (np.random.rand() < bubble_expansion_probability)
                        else: # conduit is susceptible
                            if len(conduit_neighbours[conduit]) > 0:
                                neighbour_embolization_times = embolization_times[np.array(conduit_neighbours[conduit]), 0]
                            else:
                                neighbour_embolization_times = np.array([])
                            if np.any(neighbour_embolization_times < time_step): # there are embolized neighbours
                                neighbours = conduit_neighbours[conduit]
                                embolized_neighbours = np.intersect1d(embolized_conduits, neighbours)
                                expose = False
                                if si_type == 'stochastic_sei':
                                    expose =  (np.random.rand() > (1 - spreading_probability)**(len(embolized_neighbours)))
                                elif si_type == 'physiological_sei':
                                    neighbour_bpp = np.array([conduit_neighbour_bpp[conduit][neighbour] for neighbour in embolized_neighbours])
                                    expose =  np.any(neighbour_bpp <= pressure_diff)
                                if expose:
                                    embolization_times[conduit, 2] = 0
                                    
                    if si_type in ['stochastic', 'physiological']: # SI spreading process
                        if len(conduit_neighbours[conduit]) > 0:
                            neighbour_embolization_times = embolization_times[np.array(conduit_neighbours[conduit]), 0]
                        else:
                            neighbour_embolization_times = np.array([])
                        if np.any(neighbour_embolization_times < time_step): # there are embolized neighbours
                            neighbours = conduit_neighbours[conduit]
                            embolized_neighbours = np.intersect1d(embolized_conduits, neighbours)
                            #embolize = False
                            if si_type == 'stochastic':
                                embolize =  (np.random.rand() > (1 - spreading_probability)**(len(embolized_neighbours)))
                            elif si_type == 'physiological':
                                neighbour_bpp = np.array([conduit_neighbour_bpp[conduit][neighbour] for neighbour in embolized_neighbours])
                                embolize =  np.any(neighbour_bpp <= pressure_diff)
                    if embolize:
                        n_embolized_through_spreading += 1
                        embolization_times[conduit, 0] = time_step
                        embolization_times[conduit, 2] = -1
                        if embolization_times[conduit, 1] > 0: # if conduit is functional, it will be removed from the simulation network
                            embolization_times[conduit, 1] = 0
                            conduit_pores = np.arange(conduits[conduit, 0], conduits[conduit, 1] + 1)
                            removed_conduit_indices.append(conduit)
                            pores_to_remove.extend(list(conduit_pores))
                        else: # if a nonfunctional conduit is embolized, nonfunctional component size and volume decrease
                            nonfunctional_component_size[time_step] -= 1
                            nonfunctional_component_volume[time_step] -= np.sum(np.pi * 0.5 * orig_pore_diameter[np.arange(orig_conduits[conduit, 0], orig_conduits[conduit, 1] + 1)]**2 * conduit_element_length)
            for removed_conduit_index in np.sort(np.unique(removed_conduit_indices))[::-1]:
                conduits[removed_conduit_index + 1::, 0:2] = conduits[removed_conduit_index + 1::, 0:2] - conduits[removed_conduit_index, 2]
            conduits[removed_conduit_indices, :] = -1
            if len(pores_to_remove) == perc_net['pore.coords'].shape[0]:
                last_removed_by_embolism = True
        try:
            op.topotools.trim(perc_net, pores=pores_to_remove)
            prevalence[time_step] = np.sum(embolization_times[:, 0] <= time_step) / conduits.shape[0]
            prevalence_due_to_spontaneous_embolism[time_step] = n_spontaneously_embolized / conduits.shape[0]
            prevalence_due_to_spreading[time_step] = n_embolized_through_spreading / conduits.shape[0]
            if si_type in ['stochastic_sei', 'physiological_sei']:
                fraction_of_exposed[time_step] = np.sum(embolization_times[:, 2] == 0) / conduits.shape[0]

            lcc_size[time_step], susceptibility[time_step], _ = get_conduit_lcc_size(net=perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(net=perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)

            removed_components = mrad_model.get_removed_components(perc_net, np.concatenate((conduit_elements[:,0:3]/conduit_element_length, conduit_elements[:,3::]),axis=1), cfg['net_size'][0] - 1)
            removed_elements = list(itertools.chain.from_iterable(removed_components)) # finding conduit elements in the non-functional (removed) components
            nonfunctional_component_volume[time_step] += nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * perc_net['pore.diameter'][removed_elements]**2 * conduit_element_length)
            if len(removed_elements) > 0:
                subcoords, subconns, subtypes = get_induced_subnet(perc_net, removed_elements, return_network=False)
                removed_lcc, _, n_nonfunctional_conduits = get_conduit_lcc_size(pore_coords=subcoords, conns=subconns, conn_types=subtypes, use_cylindrical_coords=use_cylindrical_coords, 
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
                
            perc_net, _ = mrad_model.clean_network(perc_net, np.concatenate((conduit_elements[:,0:3]/conduit_element_length, conduit_elements[:,3::]),axis=1), cfg['net_size'][0] - 1, remove_dead_ends=False, removed_components=removed_components)
            n_inlets[time_step], n_outlets[time_step] = get_n_inlets(perc_net, (cfg['net_size'][0] - 1)*conduit_element_length, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            mrad_model.prepare_simulation_network(perc_net, cfg, update_coords=False)
            if time_step == 0:
                water, effective_conductances[time_step], pore_pressures = simulations.simulate_water_flow(perc_net, cfg, visualize=False, return_water=True) 
            else:
                effective_conductances[time_step], pore_pressures = simulations.simulate_water_flow(perc_net, cfg, visualize=False, water=water, return_water=False)
                
            functional_lcc_size[time_step], functional_susceptibility[time_step], _ = get_conduit_lcc_size(net=perc_net)
        except Exception as e:
            if str(e) == 'Cannot delete ALL pores': # this is because all remaining nodes get embolized or belong to non-functional components
                nonfunctional_component_size[time_step::] = conduits.shape[0] - np.sum(embolization_times[:, 0] <= time_step) # all conduits that are not embolized are non-functional
                if last_removed_by_embolism:
                    nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1]
                else:
                    nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * perc_net['pore.diameter']**2 * conduit_element_length)
                lcc_size[time_step::] = max_removed_lcc
                prevalence[time_step::] = np.sum(embolization_times[:, 0] <= time_step) / conduits.shape[0]
                prevalence_due_to_spontaneous_embolism[time_step::] = n_spontaneously_embolized / conduits.shape[0]
                prevalence_due_to_spreading[time_step::] = n_embolized_through_spreading / conduits.shape[0]
                if si_type in ['stochastic_sei', 'physiological_sei']:
                    fraction_of_exposed[time_step::] = np.sum(embolization_times[:, 2] == 0) / conduits.shape[0]
                time_step += 1
                break
            elif (str(e) == "'throat.conns'") and (len(perc_net['throat.all']) == 0): # this is because all links have been removed from the network by op.topotools.trim
                nonfunctional_component_size[time_step::] = conduits.shape[0] - np.sum(embolization_times[:, 0] <= time_step)
                if last_removed_by_embolism:
                    nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1]
                else:
                    nonfunctional_component_volume[time_step::] = nonfunctional_component_volume[time_step - 1] + np.sum(np.pi * 0.5 * perc_net['pore.diameter'] * conduit_element_length)
                lcc_size[time_step::] = max_removed_lcc
                prevalence[time_step::] = np.sum(embolization_times[:, 0] <= time_step) / conduits.shape[0]
                prevalence_due_to_spontaneous_embolism[time_step::] = n_spontaneously_embolized / conduits.shape[0]
                prevalence_due_to_spreading[time_step::] = n_embolized_through_spreading / conduits.shape[0]
                if si_type in ['stochastic_sei', 'physiological_sei']:
                    fraction_of_exposed[time_step::] = np.sum(embolization_times[:, 2] == 0) / conduits.shape[0]
                time_step += 1
                break
            else:
                raise
        if time_step > si_tolerance_length:
            prevalence_diff = np.abs(prevalence[time_step] - prevalence[time_step - si_tolerance_length])
            
        time_step += 1
    
    if include_orig_values:
        effective_conductances = np.append(np.array([orig_effective_conductance]), effective_conductances[0:time_step])
        lcc_size = np.append(np.array([orig_lcc_size]), lcc_size[0:time_step])
        functional_lcc_size = np.append(np.array([orig_functional_lcc_size]), functional_lcc_size[0:time_step])
        nonfunctional_component_size = np.append(np.array([orig_nonfunctional_component_size]), nonfunctional_component_size[0:time_step])
        susceptibility = np.append(np.array([orig_susceptibility]), susceptibility[0:time_step])
        functional_susceptibility = np.append(np.array([orig_functional_susceptibility]), functional_susceptibility[0:time_step])
        n_inlets = np.append(np.array([orig_n_inlets]), n_inlets[0:time_step])
        n_outlets = np.append(np.array([orig_n_outlets]), n_outlets[0:time_step])
        nonfunctional_component_volume = np.append(np.array([orig_nonfunctional_component_volume]), nonfunctional_component_volume[0:time_step])
        prevalence = np.append(np.array([orig_prevalence]), prevalence[0:time_step])
        prevalence_due_to_spontaneous_embolism = np.append(np.array([orig_prevalence_due_to_spontaneous_embolism]), prevalence_due_to_spontaneous_embolism[0:time_step])
        prevalence_due_to_spreading = np.append(np.array([orig_prevalence_due_to_spreading]), prevalence_due_to_spreading[0:time_step])
       
        if si_type in ['stochastic_sei', 'physiological_sei']:
            fraction_of_exposed = np.append(np.array([orig_fraction_of_exposed]), fraction_of_exposed[0:time_step])
            return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading, fraction_of_exposed
        else:
            return effective_conductances, lcc_size, functional_lcc_size, nonfunctional_component_size, susceptibility, functional_susceptibility, n_inlets, n_outlets, nonfunctional_component_volume, prevalence, prevalence_due_to_spontaneous_embolism, prevalence_due_to_spreading
    elif si_type in ['stochastic_sei', 'physiological_sei']:
        return effective_conductances[0:time_step], lcc_size[0:time_step], functional_lcc_size[0:time_step], nonfunctional_component_size[0:time_step], susceptibility[0:time_step], functional_susceptibility[0:time_step], n_inlets[0:time_step], n_outlets[0:time_step], nonfunctional_component_volume[0:time_step], prevalence[0:time_step], prevalence_due_to_spontaneous_embolism[0:time_step], prevalence_due_to_spreading[0:time_step], fraction_of_exposed[0:time_step]
    else:
        return effective_conductances[0:time_step], lcc_size[0:time_step], functional_lcc_size[0:time_step], nonfunctional_component_size[0:time_step], susceptibility[0:time_step], functional_susceptibility[0:time_step], n_inlets[0:time_step], n_outlets[0:time_step], nonfunctional_component_volume[0:time_step], prevalence[0:time_step], prevalence_due_to_spontaneous_embolism[0:time_step], prevalence_due_to_spreading[0:time_step]

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
        Dc: float, average conduit diameter (m)
        Dc_cv: float, coefficient of variation of conduit diameter
        fc: float, average contact fraction between two conduits
        fpf: float, average pit field fraction between two conduits
        conduit_diameters: np.array of floats, diameters of the conduits, or 'lognormal'
        to draw diameters from a lognormal distribution defined by Dc and Dc_cv
        cec_indicator: int, value used to indicate that the type of a throat is CE
        tf: float, microfibril strand thickness (m)
        icc_length: float, length of an ICC throat (m)
        gas_contact_angle: float, the contact angle between the water and air phases (radians)
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
    cfg['gas_contact_angle'] = np.pi * cfg['gas_contact_angle'] / 180 # transformation from radians to degrees
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
    mrad_model.prepare_simulation_network(net, cfg)
    invasion_pressure = simulations.simulate_drainage(net, start_pores, cfg)
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
            mrad_model.prepare_simulation_network(perc_net, cfg)
            invasion_pressure = simulations.simulate_drainage(perc_net, start_pores, cfg)
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
            lcc_size[i], susceptibility[i], _ = get_conduit_lcc_size(net=perc_net, use_cylindrical_coords=use_cylindrical_coords, 
                                                                  conduit_element_length=conduit_element_length, 
                                                                  heartwood_d=heartwood_d, cec_indicator=cec_indicator)
            conduit_elements = mrad_model.get_conduit_elements(net=perc_net, cec_indicator=cec_indicator, 
                                                               conduit_element_length=conduit_element_length, heartwood_d=heartwood_d, use_cylindrical_coords=use_cylindrical_coords)
            orig_perc_net = op.network.Network(coords=perc_net['pore.coords'], conns=perc_net['throat.conns']) # orig_perc_net is needed for calculating measures related to non-functional components (these are removed from perc_net before the simulations)
            orig_perc_net['throat.type'] = perc_net['throat.type']
            orig_perc_net['pore.diameter'] = perc_net['pore.diameter']
            perc_net, removed_components = mrad_model.clean_network(perc_net, np.concatenate((conduit_elements[:,0:3]/conduit_element_length, conduit_elements[:,3::]),axis=1), cfg['net_size'][0] - 1, remove_dead_ends=False) # removes non-functional components from perc_net
            removed_elements = list(itertools.chain.from_iterable(removed_components)) # calculating the size of the largest removed component in conduits
            nonfunctional_component_volume[i] += nonfunctional_component_volume[i - 1] + np.sum(np.pi * 0.5 * orig_perc_net['pore.diameter'][removed_elements]**2 * conduit_element_length)
            if len(removed_elements) > 0:
                removed_net = get_induced_subnet(orig_perc_net, removed_elements, return_network=True)
                removed_lcc, _, n_nonfunctional_conduits = get_conduit_lcc_size(net=removed_net, use_cylindrical_coords=use_cylindrical_coords, 
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
            n_inlets[i], n_outlets[i] = get_n_inlets(perc_net, (cfg['net_size'][0] - 1)*conduit_element_length, cec_indicator=cec_indicator, 
                                                     conduit_element_length=conduit_element_length, heartwood_d=heartwood_d,
                                                     use_cylindrical_coords=use_cylindrical_coords)
            mrad_model.prepare_simulation_network(perc_net, cfg)
            effective_conductances[i], _ = simulations.simulate_water_flow(perc_net, cfg, visualize=False)
            functional_lcc_size[i], functional_susceptibility[i], _ = get_conduit_lcc_size(net=perc_net)
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

def get_conduit_lcc_size(net=None, pore_coords=[], conns=[], conn_types=[], use_cylindrical_coords=False, conduit_element_length=params.Lce, 
                         heartwood_d=params.heartwood_d, cec_indicator=params.cec_indicator):
    """
    Calculates the largest connected component size and susceptibility in a network where nodes correspond to conduits
    (= sets of conduit elements connected in row/z direction).

    Parameters
    ----------
    net : openpnm.Network(), optional
        pores correspond to conduit elements, throats to connections between them (default: None, in which case
        pore_coords, conns, and conn_types arrays are used to read network information)
    pore_coords : np.array(), optional
        pore coordinates of net (default: [], in which case coordinates are read from net). Note that if net is not None, pore_coords
        is not used.
    conns : np.array(), optional
        throats of net, each row containing the indices of the start and end pores of a throat (default: [], in which case throat info is read from net).
        Note that if net is not None, conns is not used.
    conn_types : np.array(), optional
        types of the throats (default: [], in which case throat info is read from net). Note that if net is not None, conn_types is not used.
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
    assert (not net is None) or (len(pore_coords) > 0), 'You must give either the net object or the pore_coords array'
    
    if not net is None:
        conduit_elements = mrad_model.get_conduit_elements(net=net, use_cylindrical_coords=use_cylindrical_coords, 
                                                           conduit_element_length=conduit_element_length, 
                                                           heartwood_d=heartwood_d, cec_indicator=cec_indicator)
        throats = net.get('throat.conns', [])
        n_pores = net['pore.coords'].shape[0]
    else:
        conduit_elements = mrad_model.get_conduit_elements(pore_coords=pore_coords, conns=conns, conn_types=conn_types, use_cylindrical_coords=use_cylindrical_coords, 
                                                           conduit_element_length=conduit_element_length, 
                                                           heartwood_d=heartwood_d, cec_indicator=cec_indicator)
        throats = conns
        n_pores = pore_coords.shape[0]
        
    throat_conduits = []
    conduit_indices = []
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
        n_conduits = n_pores
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
    conduit_elements = mrad_model.get_conduit_elements(net=net, use_cylindrical_coords=use_cylindrical_coords, 
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
    
def get_induced_subnet(net, elements, return_network=True):
    """
    Constructs the pore.coords, throat.conns, and throat.types arrays for a subgraph 
    subgraph of an op.Network spanned by a given set of pores (conduit elements) and, optionally,
    creates a new op.Network object using these arrays.

    Parameters
    ----------
    net : op.Network()
        pores correspond to conduit elements and throats to connections between them
    elements : list of ints
        the conduit elements (pores) that span the subnetwork
    return_network : bln, optional
        if True, a new op.Network object corresponding to the induced subgraph is created

    Returns
    -------
    subnet : op.Network()
        the subnetwork induced by elements
    OR
    subcoords : np.array
        pore coordinates of the induced subgraph. The size of subcoords is len(elements) x 3.
    subconns : np.array
        throats of the induced subgraph; each row of subconns contains the indices of the start and end pores
        of the throat
    subtypes : np.array
        types of the subconns
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
    if return_network:
        if subconns.shape[0] > 0:
            subnet = op.network.Network(conns=subconns, coords=subcoords)
            subnet['throat.type'] = subtypes
        else:
            subnet = op.network.Network(coords=subcoords)
        return subnet
    else: 
        return subcoords, subconns, subtypes

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
    conduit_elements = mrad_model.get_conduit_elements(net=net, use_cylindrical_coords=use_cylindrical_coords,
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
    conduit_elements = mrad_model.get_conduit_elements(net=net, use_cylindrical_coords=use_cylindrical_coords,
                                                       conduit_element_length=conduit_element_length, 
                                                       heartwood_d=heartwood_d, cec_indicator=cec_indicator)
    inlet_conduit_indices = []
    for pore in conduit_elements:
        if pore[0] == 0:
            inlet_conduit_indices.append(int(pore[3]))
    return inlet_conduit_indices

def calculate_bpp(net, conduits, icc_mask, cfg):
    """
    Draws the bubble propagation pressure (BPP) for each ICC from a Weibull distribution defined per equation 8
    of the Mrad et al. article. 

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
        conduit_element_length: float, length of a conduit element
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
        Am = 0.5 * (conduit_areas[start_conduit] / icc_count[start_conduit] + conduit_areas[end_conduit] / icc_count[end_conduit]) * fc * fpf # Mrad et al. 2018, Eq. 2
        pit_count = Am / average_pit_area # Mrad et al. 2018, Eq. 3 (using average pit area as input instead of average pit diameter)
        bpp[i] = (weibull_a / pit_count**(1 / weibull_b)) * np.random.weibull(weibull_b) # this is BPP solved from Mrad et al. equation 8
        
    return bpp

def get_spontaneous_embolism_probability(pressures):
    """
    Using the formalism of Ingram et al. 2024, calculates the probability of spontaneous embolism,
    that is, the probability of a nanobubble to expand filling its entire conduit. Probabilities are
    calculated for a set of bubble surface pressures, i.e., pressure differences between the water
    outside the bubble and air inside the bubble.

    Parameters
    ----------
    pressures : iterable of floats
        pressure, at which to calculate the probability

    Returns
    -------
    spontaneous_embolism_probabilities : dic where keys are pressure values, values probabilities (floats between 0 and 1)
        probabilities of spontaneous embolism
    """
    T = 300.0 # K
    apl = 0.6 # 'equilibrium' area per lipid, nm^-2
    mean_r = 190 # mean bubble radius, nm
    mu = np.log(mean_r)
    sigma = 0.6 # standard deviation in natural log units
    n_bubble = 5000 # number of bubbles in sample

    # lets not go crazy here
    assert mu - 3*sigma > 1

    # radii values to evaluate the Gibbs free energy at
    r_range = np.logspace(
        start=mu - 3*sigma,
        stop=mu + 6*sigma,
        base=np.e,
        num=500
        )
    
    unique_pressures = np.unique(pressures)
    embolization_probabilities = {}
    for pressure in unique_pressures:
        embolization_probability = bubble_expansion.probability(-pressure, T, mu, sigma, apl, r_range/1e9, n_bubble)
        embolization_probabilities[pressure] = embolization_probability
        
    return embolization_probabilities

