import numpy as np
import logging
from scipy.signal import find_peaks
from ..utils.hyperparameters import (
        N_tau_default, 
        reg_par_default, 
        tau_min_default,
        tau_max_default, 
        tau_sampling_default 
        )

from scipy.optimize import curve_fit

def create_basis(t, tau):
    """
    Creates the basis matrix Phi and normalizes it.
    """
    Phi = np.exp(-t[None, :] / tau[:, None])  # Shape: [N_basis, N_time]
    norms = np.linalg.norm(Phi, axis=1)
    Phi /= norms[:, None]
    return Phi, norms

def gram_matrix(Phi):
    """
    Computes the Gram matrix of the basis functions.
    """
    return np.dot(Phi, Phi.T)

def create_non_uniform_basis(time, tau_min, tau_max, N_exp):
    tau = np.logspace(np.log10(tau_min), np.log10(tau_max), N_exp)
    Phi, norms = create_basis(time, tau)
    return Phi, tau, norms

def create_uniform_basis(time, tau_min, tau_max, N_exp):
    tau = np.linspace(tau_min, tau_max, N_exp)
    Phi, norms = create_basis(time, tau)
    return Phi, tau, norms

def create_custom_basis(time, tau_candidates):
    Phi = np.exp(-time[:, None] / tau_candidates[None, :])
    norms = np.linalg.norm(Phi, axis=1)
    Phi /= norms[:, None]
    return Phi, norms

def decompose_l2_stat(Phi, signal, alpha):
    """
    Performs multiexponential decomposition using L2 regularization.
    """
    Nc, Nt = signal.shape  # Number of curves, number of time points
    A = gram_matrix(Phi) + alpha * np.identity(Phi.shape[0])
    M_l2 = np.zeros((Nc, Phi.shape[0]))
    data_fit_l2 = np.zeros((Nc, Nt))

    for i in range(Nc):
        y = signal[i, :]  # Shape: [N_time]
        b = np.dot(Phi, y)  # Shape: [N_basis]
        M_l2[i, :] = np.linalg.solve(A, b)
        data_fit_l2[i, :] = np.dot(M_l2[i, :], Phi)

    return M_l2, data_fit_l2

def perform_l2_analysis(time, data_baseline_subtracted, N_tau=N_tau_default, 
                        alpha=reg_par_default, tau_min=tau_min_default, 
                        tau_max=tau_max_default, tau_sampling=tau_sampling_default):
    if tau_sampling == 'non_uniform':
        Phi, tau, norms = create_non_uniform_basis(time, tau_min, tau_max, N_tau)
    elif tau_sampling == 'uniform':
        Phi, tau, norms = create_uniform_basis(time, tau_min, tau_max, N_tau)
    else:
        logging.warning("Unknown tau_sampling method.")
        return None, None, None, None


    M_l2, data_fit_l2 = decompose_l2_stat(Phi, data_baseline_subtracted, alpha)

    logging.info("Multiexponential analysis (L2) completed.")
    return tau, M_l2, data_fit_l2, norms

def sparcify_spectrum(tau, smooth_spectrum, min_peak_height=1e-5, norms=None):
    spectrum_squared = smooth_spectrum**2
    peaks, properties = find_peaks(spectrum_squared, height=min_peak_height)
    sparse_spectrum = np.zeros_like(smooth_spectrum)

    if len(peaks) == 0:
        logging.debug("No peaks found by sparcify_spectrum.")
        return sparse_spectrum

    boundaries = [0]
    for i in range(len(peaks)-1):
        mid_point = (peaks[i] + peaks[i+1]) // 2
        boundaries.append(mid_point)
    boundaries.append(len(smooth_spectrum)-1)

    for i, p in enumerate(peaks):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        if end_idx <= start_idx:
            continue
        hump_indices = np.arange(start_idx, end_idx+1)
        hump_values = smooth_spectrum[hump_indices]
        hump_tau = tau[hump_indices]
        total_area = np.trapz(hump_values, hump_tau)
        if norms is not None and p < len(norms):
            total_area *= norms[p]
        sparse_spectrum[p] = total_area

    return sparse_spectrum

def select_peaks(relaxation_times, amplitudes, min_peak_height=1e-5):
    if amplitudes is None or relaxation_times is None:
        logging.warning("No amplitudes or relaxation times available.")
        return None
    mean_spectrum = np.mean(amplitudes, axis=0) if amplitudes.ndim > 1 else amplitudes
    spectrum_squared = mean_spectrum**2
    peaks, properties = find_peaks(spectrum_squared, height=min_peak_height)
    if len(peaks) == 0:
        logging.warning("No significant peaks found.")
        return None
    candidate_tau = relaxation_times[peaks]
    logging.info(f"Selected {len(candidate_tau)} candidate tau values for sparse fitting.")
    return candidate_tau



# # # # # # # # # # # # # # # # # # Curve fitting 
def decompose_three_exp_nonlinear(time, data, approx_taus, fit_baseline=False):
    """
    Low-level routine for 3-exponential fitting (with optional baseline).
    
    Fits each row (curve) in 'data' separately.
    
    Parameters
    ----------
    time : 1D ndarray of shape (Nt,)
        Time axis.
    data : 2D ndarray of shape (Nc, Nt)
        Nc curves, each with Nt time points.
    approx_taus : tuple or list of length 3
        The three fixed relaxation times (tau1, tau2, tau3).
    fit_baseline : bool, optional
        If True, also fit a constant offset/baseline 'c'.
        If False, fit only the three exponential amplitudes.

    Returns
    -------
    params : 2D ndarray
        Fitted parameters for each curve.
        If fit_baseline=False, shape is (Nc, 3) => [A1, A2, A3].
        If fit_baseline=True,  shape is (Nc, 4) => [c, A1, A2, A3].
    fitted_data : 2D ndarray of shape (Nc, Nt)
        The fitted curves corresponding to each row of 'data'.
    """
    tau1, tau2, tau3 = approx_taus
    Nc, Nt = data.shape

    # Define the appropriate model
    if fit_baseline:
        def model(t, c, A1, A2, A3):
            return (c
                    + A1*np.exp(-t/tau1)
                    + A2*np.exp(-t/tau2)
                    + A3*np.exp(-t/tau3))
        param_shape = (Nc, 4)
    else:
        def model(t, A1, A2, A3):
            return (A1*np.exp(-t/tau1)
                    + A2*np.exp(-t/tau2)
                    + A3*np.exp(-t/tau3))
        param_shape = (Nc, 3)

    params = np.zeros(param_shape)
    fitted_data = np.zeros((Nc, Nt))

    for i in range(Nc):
        y = data[i, :]

        # Make a reasonable initial guess
        if fit_baseline:
            c0 = np.min(y)
            guess_amp = (np.max(y) - np.min(y)) / 3.0
            p0 = [c0, guess_amp, guess_amp, guess_amp]  # [c, A1, A2, A3]
        else:
            p0 = [np.max(y)/3.0]*3  # [A1, A2, A3]

        try:
            popt, _ = curve_fit(model, time, y, p0=p0)
            params[i, :] = popt
            fitted_data[i, :] = model(time, *popt)
        except RuntimeError:
            logging.warning(f"[decompose_three_exp_nonlinear] Fit failed for curve {i}.")
            params[i, :] = 0.0
            fitted_data[i, :] = 0.0

    return params, fitted_data


def perform_three_exp_analysis(time, data, approx_taus, fit_baseline=False):
    """
    Single high-level function for 3-exponential analysis.
    
    Depending on 'fit_baseline', either fits:
      - Three amplitudes A1, A2, A3    (if fit_baseline=False), or
      - Baseline c + A1, A2, A3        (if fit_baseline=True).

    Returns
    -------
    params : 2D ndarray
        If fit_baseline=False => shape (Nc, 3) => [A1, A2, A3].
        If fit_baseline=True  => shape (Nc, 4) => [c, A1, A2, A3].
    fitted_data : 2D ndarray of shape (Nc, Nt)
        The fitted 3-exponential (plus baseline) curves.
    """
    params, fitted_data = decompose_three_exp_nonlinear(
        time=time, 
        data=data, 
        approx_taus=approx_taus, 
        fit_baseline=fit_baseline
    )
    return params, fitted_data
