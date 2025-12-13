
"""
PS-based multi-exponential analysis (L1/L2 mixed regularisation).

Public API:
    compute_ps_l1l2(signal, t, tau_grid, lam=0.06, eps=0.1, x0=None)
    compute_ps_l2_batch(signals, t, tau_grid, alpha=1e-3, beta=1e-6)
"""
import numpy as np
from scipy import optimize

__all__ = ["compute_ps_l1l2", "compute_ps_l2_batch"]

def _phi_matrix(t, tau):
    return np.exp(-t[:, None] / tau[None, :])

def _obj_l1l2(g, Phi, y, lam, eps):
    return np.linalg.norm(Phi @ g - y) ** 2 + lam * (eps * np.linalg.norm(g) +
                                                     (1.0 - eps) * np.linalg.norm(g, 1))

def _grad_l1l2(g, Phi, y, lam, eps):
    return 2 * Phi.T @ (Phi @ g - y) + lam * ((1 - eps) * np.sign(g) +
                                              eps * g / np.linalg.norm(g))

def compute_ps_l1l2(signal, t, tau_grid, lam=0.06, eps=0.1, x0=None):
    """
    BFGS-based mixed L1/L2 optimisation for one curve.
    Returns estimated amplitudes g.
    """
    Phi = _phi_matrix(t, tau_grid)
    x0 = np.random.normal(0.0, 0.1, len(tau_grid)) if x0 is None else x0
    res = optimize.minimize(_obj_l1l2, x0, jac=_grad_l1l2,
                            args=(Phi, signal, lam, eps),
                            method="BFGS", tol=1e-9)
    return res.x

# ===== L2 batch method (аналитическая формула) =====
def _trap(h, y):
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

def _scalar_prod(f, t, tau1):
    h = t[1] - t[0]
    return _trap(h, f * np.exp(-t / tau1))

def compute_ps_l2_batch(signals, t, tau_grid, alpha=1e-3, beta=1e-6, n_iter=3000):
    """
    Implements Tikhonov + finite-difference Regularization.
    signals : ndarray (N_curves, N_points)
    t       : ndarray (N_points,)
    """
    Ne, Ntau = signals.shape[0], tau_grid.size
    F = np.empty((Ne, Ntau))
    for i in range(Ne):
        F[i] = np.array([_scalar_prod(signals[i], t, tau) for tau in tau_grid])
    G = np.array([[tau_grid[i]*tau_grid[j]/(tau_grid[i]+tau_grid[j])
                   for j in range(Ntau)] for i in range(Ntau)])
    Ginv = np.linalg.inv(G)
    lam = np.linalg.solve(G + 3*alpha*Ginv, F.T).T       # initial
    for _ in range(n_iter):
        lam_prev = lam.copy()
        # внутренний трёх-диаг. итератор
        lam[1:-1] = np.linalg.solve(G + (alpha+2*beta)*Ginv,
                                    F[1:-1] + beta * Ginv @ (lam_prev[:-2] + lam_prev[2:]).T).T
        lam[0]    = np.linalg.solve(G + (alpha+beta)*Ginv,
                                    F[0] + beta * Ginv @ lam_prev[1])
        lam[-1]   = np.linalg.solve(G + (alpha+beta)*Ginv,
                                    F[-1] + beta * Ginv @ lam_prev[-2])
    return lam
