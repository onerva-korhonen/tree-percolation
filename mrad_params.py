#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:49:37 2023

@author: onerva

Contains the xylem network parameters used in the Mrad et al. publications or their
Matlab code (https://github.com/mradassaad/Xylem_Network_Matlab)
and other default parameters for the network construction
"""
import numpy as np

net_size = np.array([11,10,14])
heartwood_d = 10 # diameter of the heartwood (in columns)

# dimensions of conduit elements
Lce = 0.00288 # length of a conduit element (m)
Dc = 24.6e-6 # averege conduit diameter (same for all elements of the conduit) (m)
Dc_cv = 0.12 # coefficient of variation of conduit diameter
Dp = 32.0e-9 # average pit membrane pore diameter (m)
Dm = 6.3e-6 # average membrane diameter (m)
fc = 0.31 # average contact fraction between two conduits
fpf = 0.7 # average pit field fraction between two conduits
Tm = 234e-9 # average membrane thickness ("Stretching") (m)
conduit_diameters = np.array([2.59695012020929e-05, 1.89719235427943e-05, 2.50441864520397e-05, 2.01174153073134e-05,
2.36521352400900e-05, 2.45039405190285e-05, 2.62275625745002e-05, 2.83605327062529e-05,
2.78028561263581e-05, 2.85128841698713e-05, 2.77499279451283e-05, 2.58281097706878e-05,
3.13732244907022e-05, 2.20945349729528e-05, 2.51375761457215e-05, 2.66829771829793e-05,
2.52659381737277e-05, 2.56500963089344e-05, 2.18724763021799e-05, 2.45908885416240e-05,
2.23506751270883e-05, 2.46952884354669e-05, 2.65910551339282e-05, 2.91177800453654e-05,
2.34667978515227e-05, 2.22271483379873e-05, 2.57657370009136e-05, 2.25081219117901e-05,
2.51420027016720e-05, 2.48594106378756e-05, 2.64581613200238e-05, 2.70316552227451e-05,
2.64857859508484e-05, 2.56149900096174e-05, 2.22965934981470e-05, 2.35580711286065e-05,
1.99663591043526e-05, 1.80735496041548e-05, 2.90701220758194e-05, 2.27183181494770e-05,
2.47643308649842e-05, 2.39978433463022e-05, 2.29116056167091e-05, 2.01444195526456e-05,
2.81223897544440e-05, 2.81207603066108e-05, 2.62314485453754e-05, 2.05319642081224e-05,
2.39807320307804e-05, 2.80415778338121e-05, 2.64374792766179e-05, 2.46968750313185e-05,
2.91852093513475e-05, 2.25302361419261e-05, 2.06405745967366e-05, 2.37927809133204e-05,
2.36576224314216e-05, 2.11720184460464e-05, 2.43505850592287e-05, 2.15147656481931e-05,
2.82081671452267e-05, 2.91960723076098e-05, 2.28001111814988e-05, 2.12284269887658e-05,
2.31055545579096e-05, 2.73019772123207e-05, 2.54575900654622e-05, 2.49295377501359e-05,
2.49924162158922e-05, 2.49054457118072e-05, 2.29697578503356e-05, 2.15572830316393e-05,
2.83988025661316e-05, 2.85801446774306e-05, 2.42167637409605e-05, 2.43037345534074e-05,
2.16318335322448e-05, 2.49697405834579e-05, 2.30691132867034e-05, 2.16709088837271e-05,
2.72386061974333e-05, 2.56287476435550e-05, 2.32780742497059e-05, 2.72622777694080e-05,
2.73704191410586e-05, 2.39992270493166e-05, 2.46867108581783e-05, 2.75365215826140e-05,
2.51083054817169e-05, 2.45249109338536e-05, 2.50905860934747e-05, 2.32785502665617e-05,
2.22240413910612e-05 ])
tf = 30e-9 # microfibril strand thickness (m)
icc_length = 1e-12 # length of an ICC throat (m)
# The two following measures are in the Mrad et al. article used to tune the Pe_rad and Pe_tan connection probabilities
GI = 0 # TODO: Fix GI (= grouping index) is the number of conduits in a cross-section divided by the number of conduit groups
conduit_density = 0 # TODO: fix density of conduits per area in cross-section

# probabilities used for network construction
Pc = np.array([0.75, 0.75])
NPc = np.array([0.75, 0.75])
Pe_rad = np.array([0.9,0.9])
Pe_tan = np.array([0.02, 0.02])
weibull_a = 20.28E6 # Weibull distribution scale parameter, Pa
weibull_b = 3.2 # Weibull distribution shape parameter

# the numbers used to seed the random number generation for initiating and
# terminating conduits in the Mrad article
seeds_NPc = [205, 9699, 8324, 2123, 1818, 1834, 3042, 5247, 4319, 2912]
seeds_Pc = [6118, 1394, 2921, 3663, 4560, 7851, 1996, 5142, 5924,  464]
seed_ICC_rad = 63083
seed_ICC_tan = 73956

# throat-type indicators
icc_indicator = 100
cec_indicator = 1000

# water properties
water_pore_viscosity = 1.002e-3
water_throat_viscosity = 1.002e-3
water_pore_diffusivity = 1.0e-9
water_surface_tension = 0.072 # Pa.m

# simulation parameters
inlet_pressure = 0.1E6 # Pa 
outlet_pressure = 0E6 # Pa