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

# TODO: write functions for removing network nodes in a given order (can be random). The idea would be to do this separately
# (but in the same order) for the OpenPNM network (and calculate effective conductance after every removal) and a networkx
# graph (and calculate the largest connected componen size)

# Question: what about pore order? in which order are the openpnm pores and how to refer to them? op.create_adjacency_matrix
# creates an adjacency matrix that can be transformed into nx.Graph with nx.from_scipy_sparse_array, but how to link the
# nodes to the op pores to ensure that same pores get removed?
# for op: op.topotools.trim to remove pores/throats (takes pore/throat indices as input)

def run_op_percolation(net, conduit_elements_rows, cfg, removal_order='random'):
    """
    Removes pores from an OpenPNM network object in a given (or random) order and calculates the effective conductance
    after each removal.

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
    removal_order : iterable or str, optional (default='random')
        indices of pores to be removed in the order of removal. for performing full percolation analysis, len(removal_order)
        should match the number of pores. string 'random' can be given to indicate removal in random order.

    Returns
    -------
    effective_conductances : np.array
        effective conductance of the network after each removal
    """
    n_pores = net['pore.coords'].shape[0]
    conduit_diameters = cfg.get('conduit_diameters', params.conduit_diameters)
    cec_indicator = cfg.get('cec_indicator', params.cec_indicator)
    
    effective_conductances = np.zeros(n_pores)
    if removal_order == 'random':
        removal_order = np.arange(0, n_pores)
        np.random.shuffle(removal_order)
    for i, _ in enumerate(removal_order):
        if not isinstance(conduit_diameters, str):
            conduit_diameters_temp = np.copy(conduit_diameters)
            np.delete(conduit_diameters_temp, removal_order[:i + 1])
        sim_net = op.network.Network(conns=net['throat.conns'], coords=net['pore.coords'])
        sim_net['throat.type'] = net['throat.type']
        try:
            op.topotools.trim(sim_net, pores=removal_order[:i + 1])
            conduit_elements = mrad_model.get_conduit_elements(sim_net, cec_indicator=cec_indicator)
            sim_net = mrad_model.clean_network(sim_net, conduit_elements, cfg['net_size'][0] - 1, remove_dead_ends=False)
            sim_net = mrad_model.prepare_simulation_network(sim_net, cfg)
            effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
        except:
            effective_conductances[i] = 0
    return effective_conductances
