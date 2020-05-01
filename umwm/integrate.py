import matplotlib.pyplot as plt
import numpy as np
from umwm.dynamics import advect
from umwm.diagnostics import significant_wave_height, form_drag, \
                             mean_wave_period, dominant_wave_period
from umwm.physics import mean_squared_slope, source_input, source_dissipation, \
                         source_wave_interaction

def integrate(Fk_init, f, k, cp, cg, x, wspd, duration, dt, mss_coefficient=120, snl_coefficient=1):
    time = []
    num_time_steps = int(duration / dt)
    num_grid_points = k.shape[0]
    
    swh = np.zeros((num_time_steps, num_grid_points))
    mwp = np.zeros((num_time_steps, num_grid_points))
    dwp = np.zeros((num_time_steps, num_grid_points))
    mss = np.zeros((num_time_steps, num_grid_points))
    tau = np.zeros((num_time_steps, num_grid_points))
    
    dk = 2 * np.pi * f / cg
    Fk = 1. * Fk_init[:]
    
    for n in range(num_time_steps):
        time.append(n * dt)
        print(n, '/', num_time_steps)
        
        Sin = source_input(wspd, f, k, cp)
        Sds = source_dissipation(Fk, f, k, dk, mss_coefficient=mss_coefficient)
        Snl = source_wave_interaction(Fk, k, dk, snl_coefficient=snl_coefficient)
        adv = advect(Fk, cg, x)
        Fk *= np.exp(dt * (Sin - Sds))
        Fk += dt * Snl
        Fk -= dt * adv
        
        swh[n,:] = significant_wave_height(Fk, dk)
        mwp[n,:] = mean_wave_period(Fk, f)
        dwp[n,:] = dominant_wave_period(Fk, f)
        mss[n,:] = mean_squared_slope(Fk, k, dk)
        tau[n,:] = form_drag(Sin, Fk, cp, dk)

        if n % 100 == 0:
            plt.semilogx(k[0,:], Fk[-1,:], marker='.', color='k')
            plt.semilogx(k[0,:], Fk[-1,:] * source_input(10., f, k, cp)[-1,:], marker='.')
            plt.semilogx(k[0,:], Fk[-1,:] * source_dissipation(Fk, f, k, dk)[-1,:], marker='.')
            plt.semilogx(k[0,:], source_wave_interaction(Fk, k, dk)[-1,:], marker='.')
            plt.xlim(1e-1, 5e2)
            plt.xlabel('Wavenumber [rad/m]')
            plt.ylabel(r'$F_k$')
            plt.savefig('spectrum_%6.6i.png' % n)
            plt.close()
        
    time = np.array(time)
    return time, swh, mwp, dwp, mss, tau, Fk
