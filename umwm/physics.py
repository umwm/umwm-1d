import numpy as np


def frequency_logspace(fmin: float, fmax: float, num_frequencies: int) -> np.ndarray:
    """Returns an array of frequencies distributed in log-space
    in the range of [fmin, fmax] with num_frequencies increments."""
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_frequencies, endpoint=True)


def wavenumber(frequency, water_depth, grav=9.8, surface_tension=0.074, water_density=1e3):
    """Solves the dispersion relationship for wavenumber 
    using a Newton-Raphson iteration method."""
    frequency_nondim = 2 * np.pi * np.sqrt(water_depth / grav) * frequency
    k = frequency_nondim**2
    surface_tension_nondim = surface_tension / (grav * water_density * water_depth**2)
    count = 0
    while True:
        t = np.tanh(k)
        dk = - (frequency_nondim**2 - k * t * (1 + surface_tension_nondim * k**2)) \
           / (3 * surface_tension_nondim * k**2 * t + t + k * (1 + surface_tension_nondim * k**2) * (1 - t**2))
        k -= dk
        count += 1
        if count > 100:
            break
    return k / water_depth


def phase_speed(frequency, wavenumber):
    """Returns the phase speed of a wave given input frequency and wavenumber."""
    return 2 * np.pi * frequency / wavenumber


def group_speed(frequency, wavenumber, water_depth, water_density=1e3, surface_tension=0.074, grav=9.8):
    """Returns the group speed of a wave given input frequency, wavenumber, and water depth."""
    cp = phase_speed(frequency, wavenumber)
    kd = wavenumber * water_depth
    sigma_k2 = surface_tension * wavenumber**2
    return cp * (0.5 + kd / np.sinh(kd) + sigma_k2 / (water_density * grav * sigma_k2))


def source_input(wind_speed, frequency, wavenumber, phase_speed, sheltering_coefficient=0.11,
                 air_density=1.2, water_density=1e3, current=0, grav=9.8):
    """Wind input source function based on Jeffreys's sheltering hypothesis."""
    wind_speed_relative = wind_speed - phase_speed - current
    s_in = sheltering_coefficient * wind_speed_relative * np.abs(wind_speed_relative)
    s_in *= air_density / water_density * 2 * np.pi * frequency * wavenumber / grav
    return s_in


def mean_squared_slope(spectrum_k, k, dk):
    return np.sum(spectrum_k * k**2 * dk, axis=1)


def mean_squared_slope_long(spectrum_k, k, dk):
    return np.cumsum(spectrum_k * k**2 * dk, axis=1)


def saturation_spectrum(spectrum_k, k):
    return spectrum_k * k**4


def source_dissipation(spectrum_k, f, k, dk, dissipation_coefficient=42, dissipation_power=2.4, mss_coefficient=120):
    omega = 2 * np.pi * f
    if mss_coefficient > 0:
        mss = mean_squared_slope_long(spectrum_k, k, dk)
    else:
        mss = np.zeros(k.shape)
    mss_effect = (1 + mss_coefficient * mss)**2
    B_k = saturation_spectrum(spectrum_k, k)
    return dissipation_coefficient * omega * mss_effect * B_k**dissipation_power


def wind_wave_balance(source_input, frequency, wavenumber, dissipation_coefficient=42, 
                      dissipation_power=2.4, mss=0, mss_coefficient=120):
    """Returns the spectrum for which the dissipation is balanced by input."""
    omega = 2 * np.pi * frequency
    mss_effect = (1 + mss_coefficient*mss)**2
    Fk = (source_input / (omega * dissipation_coefficient * mss_effect))**(1 / dissipation_power) / wavenumber**4
    Fk[np.isnan(Fk)] = 0
    return Fk


def source_wave_interaction(spectrum_k, k, dk, snl_coefficient=1):
    """A simple, minimalisting, nonlinear downshifting function."""
    Snl = np.zeros((k.shape))
    Snl[:,:-1] = snl_coefficient * np.diff(spectrum_k, axis=1) / dk[:,1:]
    return Snl
