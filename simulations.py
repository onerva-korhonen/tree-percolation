#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:48:47 2024

@author: onervak

Contains functions for hydraulics simulations
"""
import numpy as np
import openpnm as op

import mrad_params
import params
import visualization

def simulate_water_flow(sim_net, cfg, visualize=False, water=None, return_water=False):
    """
    Using OpenPNM tools, performs Stokes flow simulation and a simple advection-diffusion simulation
    and calculates the effective conductance based on the Stokes flow simulation.

    Parameters
    ----------
    sim_net : openpnm.Network()
        pores correspond to conduit elements and throats to connections between them
    cfg : dic
        contains:
        water_pore_viscosity: float, value of the water viscosity in pores
        water_throat_viscosity: float, value of the water viscosity in throats
        water_pore_diffusivity: float, value of the water diffusivity in pores
        inlet_pressure: float, pressure at the inlet conduit elements (Pa)
        outlet_pressure: float, pressure at the outlet conduit elements (Pa) 
        Dp: float, average pit membrane pore diameter (m)
        Tm: float, average thickness of membranes (m)
        cec_indicator: int, value used to indicate that the type of a throat is CE
        use_cylindrical_coords: bln, should Mrad model coordinates be interpreted as cylindrical ones in visualizations?
    visualize : bln, optional
        if True, the pressure and concentration at pores is visualized in form of scatter plots.
    water : op.Water(), optional
        the Water() object used to simulate the flow. default: None, in which case a new object is created.
    return_water : bln, optional
        if True, the created Water() object is returned. default: False

    Returns
    -------
    effective_conductance : float
        effective conductance, i.e. total flow out of the xylem normalized by the pressure difference between inlet and outlet. 
    pore_pressure : np.array of floats
        water pressure at each pore of sim_net
    """
    water_pore_viscosity = cfg.get('water_pore_viscosity', mrad_params.water_pore_viscosity)
    water_throat_viscosity = cfg.get('water_throat_viscosity', mrad_params.water_throat_viscosity)
    water_pore_diffusivity = cfg.get('water_pore_diffusivity', mrad_params.water_pore_diffusivity)
    inlet_pressure = cfg.get('inlet_pressure', mrad_params.inlet_pressure)
    outlet_pressure = cfg.get('outlet_pressure', mrad_params.outlet_pressure)
    Dp = cfg.get('Dp', mrad_params.Dp)
    Tm = cfg.get('Tm', mrad_params.Tm)
    cec_indicator = cfg.get('cec_indicator', mrad_params.cec_indicator)
    use_cylindrical_coords = cfg.get('use_cylindrical_coords', True)
    
    n_pores = sim_net['pore.coords'].shape[0]
    conn_types = sim_net['throat.type']
    n_throats = conn_types.shape[0]
    cec_mask = conn_types == cec_indicator # cec_mask == 1 if the corresponding throat is a connection between two elements in the same conduit
    
    if water is None:
        water = op.phase.Water(network=sim_net)
        water['pore.viscosity'] = water_pore_viscosity
        water['throat.viscosity'] = water_throat_viscosity
        water['pore.diffusivity'] = water_pore_diffusivity
    
    water.add_model(propname='throat.diffusive_conductance',
                    model=op.models.physics.diffusive_conductance.ordinary_diffusion)
    water.add_model(propname='throat.hydraulic_conductance',
                    model=op.models.physics.hydraulic_conductance.generic_hydraulic) # This is the Hagen-Pouseuille equation
    water.add_model(propname='throat.ad_dif_conductance',
                    model=op.models.physics.ad_dif_conductance.ad_dif)
    pit_conductance = (Dp**3 / (24 * water['throat.viscosity']) * (1 + 16 * Tm / (3 * np.pi * Dp))**(-1) * sim_net['throat.npore'])
    water['throat.hydraulic_conductance'][~cec_mask] = pit_conductance[~cec_mask] #Set the separately calculated values for the hydraulic conductance of the ICCs
    water.regenerate_models(propnames='throat.ad_dif_conductance') # redefining the diffusional conductance of the CECs
    
    if use_cylindrical_coords:
        axnum = 2 # row information is in the last (z) column
    else:
        axnum = 0 # row information is in the first column
        
    inlet = sim_net['pore.coords'][:, axnum] == np.min(sim_net['pore.coords'][:, axnum])
    outlet = sim_net['pore.coords'][:, axnum] == np.max(sim_net['pore.coords'][:, axnum])
    
    # Stokes flow simulation
    
    stokes_flow = op.algorithms.StokesFlow(network=sim_net, phase=water,)
    stokes_flow.set_value_BC(pores=inlet, values=inlet_pressure)
    stokes_flow.set_value_BC(pores=outlet, values=outlet_pressure)
    stokes_flow.run()
    
    water['pore.pressure'] = stokes_flow['pore.pressure'] #The results calculated in the Stokes flow simulation are used in the determination of the advective-diffusive conductance in the advection-diffusion simulation
    water.regenerate_models(propnames='throat.ad_dif_conductance')
    
    if visualize:
        # visualizing water pressure at pores
        visualization.make_colored_pore_scatter(sim_net, water['pore.pressure'], title='Pressure distribution')
    
    # Steady-state advection-diffusion simulation with constant boundary values
    advection_diffusion = op.algorithms.AdvectionDiffusion(network=sim_net, phase=water)
    advection_diffusion.set_value_BC(pores=inlet, values=inlet_pressure)
    advection_diffusion.set_value_BC(pores=outlet, values=outlet_pressure)
    advection_diffusion.settings['pore_volume'] = 'pore.effective_volume'
    
    advection_diffusion.run()
    concentration = advection_diffusion['pore.concentration']
    
    if visualize:
        # visualizing concentration at pores
        visualization.make_colored_pore_scatter(sim_net, concentration, title='Concentration')
    effective_conductance = stokes_flow.rate(pores=outlet)[0] / (inlet_pressure - outlet_pressure)
    effective_conductance = effective_conductance * -1 # stokes_flow.rate gives the flow exiting the pores (flow_out - flow_in). Since the outlet pores are the edge of the xylem, there is no flow out of them, and the stokes_flow.rate(pores=outlet) is thus negative.
    
    pore_pressure = stokes_flow['pore.pressure']
    
    if return_water:
        return water, effective_conductance, pore_pressure
    else:
        return effective_conductance, pore_pressure

def simulate_drainage(sim_net, start_pores, cfg):
    """
    Simulates the invasion of air in a water-filled pore network using the openpnm Drainage algorithm. The invasion
    pressure is defined as a) inf for pores and throats not connected to start_pores, b) as the smallest 
    frequence of the given frequency range that is larger than the pore's or throut's entry pressure for
    pores and throats connected to start pores.
    
    Parameters:
    -----------
    sim_net : openpnm.Network()
        pores correspond to conduit elements and throats to connections between them
    start_pores : np.array of ints
        pores from which the drainage starts
    cfg : dic
        contains:
        gas_contact_angle : float, the contact angle between the water and air phases (degrees)
        surface_tension : float, the surface tension betweent he water and air phases (Newtons/meter)
        pressure : int or array-like, the frequency steps to investigate (if an int is given, a log-range with the corresponding number of steps is used)
        
    Returns:
    -------
    invasion_pressure : np.array of floats
        the pressure at which each pore gets invaded by air (embolized)
    """
    pressure = cfg.get('pressure', params.pressure)
    # creating the air phase
    air = op.phase.Air(network=sim_net)
    air['pore.contact_angle'] = cfg.get('gas_contact_angle')
    air['pore.surface_tension'] = cfg.get('surface_tension', params.surface_tension)
    f = op.models.physics.capillary_pressure.washburn
    air.add_model(propname = 'throat.entry_pressure', model = f, surface_tension = 'throat.surface_tension',
              contact_angle = 'throat.contact_angle', diameter = 'throat.diameter')
    air.add_model(propname='pore.entry_pressure',
                  model=f, contact_angle='pore.contact_angle', surface_tension='pore.surface_tension', diameter='pore.diameter')
    
    # creating and running the drainage algorithm
    drn = op.algorithms.Drainage(network = sim_net, phase = air)
    drn.set_inlet_BC(pores = start_pores) # NOTE: start_pores is used only to set invasion pressure of pores not connected to the seed to Inf; the invasion pressure of start_pores is not initialized to the smallest pressure investigated
    drn.run(pressure)
    
    # obtaining the invasion pressure
    invasion_pressure = drn['pore.invasion_pressure']
    
    return invasion_pressure

    

