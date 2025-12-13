
import numpy as np

__all__ = ["sin_79hz", "phi_window"]

OMEGA_79 = 79 * 2 * np.pi

def sin_79hz(t, A, phi, C):
    """79 Hz sine used in MRX fitting."""
    return A * np.sin(OMEGA_79 * t + phi) + C

def phi_window(t, A):
    """Piecewise-parabolic window phi(t, A) from оригинального скрипта."""
    if np.abs(t) > A:
        return 0.0
    return -6*t*(t + A)/A**3 if t < 0 else 6*t*(t - A)/A**3
