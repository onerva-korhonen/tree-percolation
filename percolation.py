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

# TODO: write functions for removing network nodes in a given order (can be random). The idea would be to do this separately
# (but in the same order) for the OpenPNM network (and calculate effective conductance after every removal) and a networkx
# graph (and calculate the largest connected componen size)

# Question: what about pore order? in which order are the openpnm pores and how to refer to them? op.create_adjacency_matrix
# creates an adjacency matrix that can be transformed into nx.Graph with nx.from_scipy_sparse_array, but how to link the
# nodes to the op pores to ensure that same pores get removed?
# for op: op.topotools.trim to remove pores/throats (takes pore/throat indices as input)

def run_op_percolation(net, cfg, removal_order='random'):
    """
    Removes pores from an OpenPNM network object in a given (or random) order and calculates the effective conductance
    after each removal.

    Parameters
    ----------
    net : openpnm.Network()
        pores correspond to conduit elements, throats to connections between the elements
    cfg : dict
        contains:
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
    effective_conductances = np.zeros(n_pores)
    if removal_order == 'random':
        removal_order = np.random.shuffle(np.arange(0, n_pores))
    for i, pore_to_remove in enumerate(removal_order):
        op.topotools.trim(net, pores=[pore_to_remove])
        sim_net = mrad_model.prepare_simulation_network(net, cfg)
        effective_conductances[i] = mrad_model.simulate_water_flow(sim_net, cfg, visualize=False)
    return effective_conductances
