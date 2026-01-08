#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:21:16 2025

@author: fhc986

Vessel anatomy and pit membrane parameters for Fagus sylvatica
"""
import params

import numpy as np

# probabilities for network construction
NPc = np.array([0.7, 0.7]) # probability to start a new conduit
Pc = np.array([0.9, 0.9]) # probability to end an existing conduit
Pe_rad = np.array([0.2, 0.2]) # probability of radial connection
Pe_tan = np.array([0.8, 0.8]) # probability of tangential connection

# anatomical and physiological parameters
Dp = 10E-9 # m; average pit membrane pore diameter; from Kaack et al., 2021
tf = 20E-9 # m; microfibril strand thickness; from Kaack et al., 2021
Tm = 234E-9 # m; pit membrane thickness; from Kaack et al., 2021
Lce = 357.22E-6 # m; vessel element length; from Karimi 2014
n_constrictions = int(np.floor((Tm*1E9 + 20) / 30)) # calculated following the caption of Table 1 in Kaack et al. 2021 but assuming 10 nm between microfibril strands instead of 20 nm
truncnorm_center = 10.E-9 # m; center value of truncated normal distribution; from Kaack et al. 2021, New Phytologist
truncnorm_std = 7.5E-9 # m, standard deviation of truncated normal distribution; from Kaack et al. 2021, New Phytologist
truncnorm_a = 2.5E-9 # m, beginning of the left truncation; from Kaack et al. 2021, New Phytologist
pore_shape_correction = 0.5 # factor for correcting the assumption of round shape of all pores; from Kaack et al. 2021, New Phytologist
gas_contact_angle = 0 # radians; the contact angle between gas and xylem sap; from Kaack et al. 2021, New Phytologist

segment_names = ['M42', 'M23', 'M39', 'M41', 'M32', 'M31', 'M100', 'M101', 'M24', 'M102'] # NOTE: keep all segment properties in the same order
Dc = np.array([2.980032051282051e-05, 2.3741391359593392e-05, 2.827737676056338e-05, 2.8415688073394494e-05,
      2.646912296564195e-05, 2.730665898617512e-05, 3.3230200000000006e-05, 3.358575714285714e-05,
      2.566723716381418e-05, 3.131887323943661e-05])
Dc_std = np.array([1.0585625075100896e-05, 7.422736160256487e-06, 8.986635577283445e-06, 1.0588179709436388e-05,
          8.545644928363257e-06, 9.522619016794125e-06, 1.2722401534975338e-05, 1.2997184612650007e-05,
          8.67623321298386e-06, 1.3539058838167754e-05])
Dc_cv = Dc_std / Dc
fc = [0.0183653508974359, 0.05784052033163914, 0.03685643433450705, 0.01847648837876802,
      0.031585857980108496, 0.03448462905069125, 0.03512025761, 0.03637043749,
      0.03596413490709047, 0.03554364860362172]
fpf = [0.5354785240000001, 0.5415683649999999, 0.629453652, 0.42283072,
       0.439700342, 0.6480547249999999, 0.60086713, 0.647937014,
       0.508086274, 0.5789179089999998]
karimi_fc = 0.12 # from Karimi 2014

# visualization parameters
spreading_simulation_data_stem = params.optimized_spreading_probability_save_path_base.rsplit('/', 1)[0]
spreading_simulation_data_folders = [spreading_simulation_data_stem + '/' + segment_name for segment_name in segment_names]
    # the following line style lists are from matplotlib documentation
linestyle_str = [
     'solid',      # Same as (0, ()) or '-'
     'dotted',    # Same as ':'
     'dashed',    # Same as '--'
     'dashdot']  # Same as '-.'

linestyle_tuple = [
     (0, (1, 10)),
     (0, (1, 5)),
     (0, (1, 1)),
     (5, (10, 3)),
     (0, (5, 10)),
     (0, (5, 5)),
     (0, (5, 1)),
     (0, (3, 10, 1, 10)),
     (0, (3, 5, 1, 5)),
     (0, (3, 1, 1, 1)),
     (0, (3, 5, 1, 5, 1, 5)),
     (0, (3, 10, 1, 10, 1, 10)),
     (0, (3, 1, 1, 1, 1, 1))]

linestyles = linestyle_str[::-1] + linestyle_tuple[::-1]
vc_linestyles = linestyles[:len(segment_names)]
pressures_to_be_visualized = np.concatenate((np.zeros(1), np.arange(0.5, 3.0, step=0.05)))*1E6

