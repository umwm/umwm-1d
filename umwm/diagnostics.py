import numpy as np

def significant_wave_height(spectrum_k, dk):
    return 4 * np.sqrt(np.sum(spectrum_k * dk, axis=1))


def mean_wave_period(spectrum_k, f):
    return np.sum(spectrum_k, axis=1) / np.sum(spectrum_k * f, axis=1)


def dominant_wave_period(spectrum_k, f):
    return np.sum(spectrum_k**4, axis=1) / np.sum(spectrum_k**4 * f, axis=1)


def form_drag(source_input, spectrum_k, phase_speed, dk, water_density=1e3, grav=9.8):
    return water_density * grav * np.sum(source_input * spectrum_k * dk / phase_speed, axis=1)
