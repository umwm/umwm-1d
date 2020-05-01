import matplotlib
matplotlib.rc('font', size=16)

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from umwm.integrate import integrate
from umwm.physics import frequency_logspace, wavenumber, phase_speed, \
                         group_speed, source_input, wind_wave_balance

fmin = 0.1 # Hz
fmax = 20 # Hz
num_frequencies = 50

xmin = 0 # m
xmax = 10 # m
num_grid_points = 11

water_depth = 1e3

f = frequency_logspace(fmin, fmax, num_frequencies)
x = np.linspace(xmin, xmax, num_grid_points, endpoint=True)
f, x = np.meshgrid(f, x)
depth = water_depth * np.ones(x.shape)
k = wavenumber(f, depth)
cp = phase_speed(f, k)
cg = group_speed(f, k, depth)
dk = 2 * np.pi * f / cg

Fk_init = wind_wave_balance(source_input(0.8, f, k, cp), f, k)
time, swh, mwp, dwp, mss, tau, Fk = integrate(Fk_init, f, k, cp, cg, x, 5, 3600, 0.01)
