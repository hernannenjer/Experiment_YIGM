import numpy as np
import logging
from scipy import signal
from numpy.polynomial import polynomial as poly
from scipy.interpolate import UnivariateSpline

def subtract_baseline_mean_end(raw_data, n_points=100):
    if raw_data is None:
        logging.warning("Data not loaded.")
        return None
    baseline = np.mean(raw_data[:, -n_points:], axis=1)
    data_baseline_subtracted = raw_data - baseline[:, np.newaxis]
    logging.info("Baseline (mean_end) subtracted.")
    return data_baseline_subtracted, baseline

def subtract_baseline_median(raw_data, n_points=500):
    if raw_data is None:
        logging.warning("Data not loaded.")
        return None
    baseline = np.median(raw_data[:, n_points:], axis=1)
    data_baseline_subtracted = raw_data - baseline[:, np.newaxis]
    logging.info("Baseline (median) subtracted.")
    return data_baseline_subtracted, baseline

def subtract_baseline_polynomial(raw_data, time, order=1):
    if raw_data is None or time is None:
        logging.warning("Data or time axis not loaded.")
        return None
    baselines = []
    adjusted_data = np.zeros_like(raw_data)
    for i in range(raw_data.shape[0]):
        coeffs = poly.polyfit(time, raw_data[i], order)
        baseline = poly.polyval(time, coeffs)
        baselines.append(baseline)
        adjusted_data[i] = raw_data[i] - baseline
    logging.info("Baseline (polynomial) subtracted.")
    return adjusted_data, np.array(baselines)

def subtract_baseline_lowpass(raw_data, cutoff_freq=0.1, fs=1.0):
    if raw_data is None:
        logging.warning("Data not loaded.")
        return None
    sos = signal.butter(1, cutoff_freq, 'low', fs=fs, output='sos')
    baseline = signal.sosfilt(sos, raw_data, axis=1)
    data_baseline_subtracted = raw_data - baseline
    logging.info("Baseline (lowpass) subtracted.")
    return data_baseline_subtracted, baseline

def subtract_baseline_3points(raw_data):
    """
    Subtract baseline using the 3-points method described.

    The idea:
    1) Take the second half of each signal (tail region) where all but the 
       slowest (longest-lived) exponential are assumed to have decayed.
    2) Define three subregions within that tail, and compute their mean values:
       Y1, Y2, Y3.
    3) Compute a scalar baseline from these three means via:
         baseline = (Y1 * Y3 - Y2^2) / (Y1 + Y3 - 2 * Y2)
    4) Subtract that scalar baseline from the entire signal.

    Parameters
    ----------
    raw_data : ndarray
        Shape (n_signals, n_samples). Each row is one signal.

    Returns
    -------
    data_baseline_subtracted : ndarray
        The baseline-subtracted data, shape (n_signals, n_samples).
    baseline : ndarray
        The scalar baseline for each row, shape (n_signals,).
    """
    if raw_data is None:
        logging.warning("Data not loaded.")
        return None

    # Number of samples in the full signal
    Nt = raw_data.shape[1]

    # We'll look at the second half: from Nt/2 to Nt
    #  -> shape becomes (n_signals, NV)
    V = raw_data[:, int(Nt/2):]
    NV = V.shape[1]

    # We'll define a small window eps around three points in this tail
    eps = int(NV / 10)

    # Let t1, t2, t3 be indices within this tail region
    #   e.g., near 10% of the tail, near 50% of the tail, near 2*(that 50% index)
    t1 = eps
    t2 = int(NV / 2) - eps
    t3 = int(2 * t2)

    # Means around each of the three anchor points
    Y1 = np.mean(V[:, t1 - eps : t1 + eps], axis=1)
    Y2 = np.mean(V[:, t2 - eps : t2 + eps], axis=1)
    Y3 = np.mean(V[:, t3 - eps : t3 + eps], axis=1)

    # Baseline formula
    baseline = (Y1 * Y3 - Y2**2) / (Y1 + Y3 - 2 * Y2)

    # Subtract this scalar baseline from the entire signal
    data_baseline_subtracted = raw_data - baseline[:, np.newaxis]

    logging.info("Baseline (3-points method) subtracted.")
    return data_baseline_subtracted, baseline

def subtract_baseline_scalar(raw_data, scalar):
    if raw_data is None:
        logging.warning("Data not loaded.")
        return None
    data_baseline_subtracted = raw_data - scalar
    logging.info(f"Baseline (scalar {scalar}) subtracted.")
    return data_baseline_subtracted, scalar

def denoise_spline_interpolation(data_baseline_subtracted, time, s=None):
    if data_baseline_subtracted is None:
        logging.warning("Baseline-subtracted data not available.")
        return None
    data_denoised = np.zeros_like(data_baseline_subtracted)
    for i in range(data_baseline_subtracted.shape[0]):
        spline = UnivariateSpline(time, data_baseline_subtracted[i], s=s)
        data_denoised[i] = spline(time)
    logging.info("Data denoised (spline).")
    return data_denoised
