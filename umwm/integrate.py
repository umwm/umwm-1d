import matplotlib.pyplot as plt
import numpy as np
from umwm.dynamics import advect
from umwm.diagnostics import significant_wave_height, form_drag, \
                             mean_wave_period, dominant_wave_period
from umwm.physics import mean_squared_slope, source_input, source_dissipation, \
                         source_wave_interaction

def integrate(Fk_init, f, k, cp, cg, x, wspd, duration, output_interval,
              mss_coefficient=120, snl_coefficient=1):
    num_time_steps = int(duration / output_interval)
    num_grid_points = k.shape[0]
    
    time = np.zeros((num_time_steps))
    swh = np.zeros((num_time_steps, num_grid_points))
    mwp = np.zeros((num_time_steps, num_grid_points))
    dwp = np.zeros((num_time_steps, num_grid_points))
    mss = np.zeros((num_time_steps, num_grid_points))
    tau = np.zeros((num_time_steps, num_grid_points))
    
    dk = 2 * np.pi * f / cg
    Fk = 1. * Fk_init[:]
   
    for n in range(num_time_steps):
        elapsed = 0
        print('integrate: time step ', n, '/', num_time_steps)
        while elapsed < output_interval:
        
            Sin = source_input(wspd, f, k, cp)
            Sds = source_dissipation(Fk, f, k, dk, mss_coefficient=mss_coefficient)
            Snl = source_wave_interaction(Fk, k, dk, snl_coefficient=snl_coefficient)

            dt = np.min([0.1 / np.max(np.abs(Sin - Sds)), output_interval - elapsed])

            Fk = Fk * np.exp(dt * (Sin - Sds)) + dt * (Snl - advect(Fk, cg, x))
            elapsed += dt
        
        time[n] = n * output_interval
        swh[n,:] = significant_wave_height(Fk, dk)
        mwp[n,:] = mean_wave_period(Fk, f)
        dwp[n,:] = dominant_wave_period(Fk, f)
        mss[n,:] = mean_squared_slope(Fk, k, dk)
        tau[n,:] = form_drag(Sin, Fk, cp, dk)
        
    return time, swh, mwp, dwp, mss, tau, Fk
