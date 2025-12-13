"""
Analysis Strategy Pattern Implementation
Provides a unified interface for different multi-exponential analysis methods.
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any

from ..utils.hyperparameters import (
    N_tau_default, 
    reg_par_default,
    reg_par_ps_default,
    tau_min_default, 
    tau_max_default, 
    tau_sampling_default
)


class AnalysisStrategy(ABC):
    """
    Abstract base class for analysis strategies.
    """
    
    @abstractmethod
    def analyze(self, time: np.ndarray, data: np.ndarray, **kwargs) -> Tuple:
        """
        Perform analysis on the given data.
        
        Parameters
        ----------
        time : np.ndarray
            Time axis
        data : np.ndarray
            Data to analyze (N_curves x N_time)
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        tuple
            (tau, amplitudes, fitted_data, additional_info)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the analysis method."""
        pass


class L2AnalysisStrategy(AnalysisStrategy):
    """
    L2 regularization (Ridge regression) analysis strategy.
    """
    
    def __init__(
        self,
        N_tau: int = N_tau_default,
        alpha: float = reg_par_default,
        tau_min: float = tau_min_default,
        tau_max: float = tau_max_default,
        tau_sampling: str = tau_sampling_default
    ):
        self.N_tau = N_tau
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_sampling = tau_sampling
    
    def analyze(self, time: np.ndarray, data: np.ndarray, **kwargs) -> Tuple:
        """Perform L2 analysis."""
        from .me_analysis import perform_l2_analysis
        
        return perform_l2_analysis(
            time=time,
            data_baseline_subtracted=data,
            N_tau=kwargs.get('N_tau', self.N_tau),
            alpha=kwargs.get('alpha', self.alpha),
            tau_min=kwargs.get('tau_min', self.tau_min),
            tau_max=kwargs.get('tau_max', self.tau_max),
            tau_sampling=kwargs.get('tau_sampling', self.tau_sampling)
        )
    
    def get_name(self) -> str:
        return "L2 (Ridge)"


class ElasticNetAnalysisStrategy(AnalysisStrategy):
    """
    Elastic Net (L1 + L2) regularization analysis strategy.
    """
    
    def __init__(
        self,
        N_tau: int = N_tau_default,
        alpha: float = reg_par_ps_default,
        tau_min: float = tau_min_default,
        tau_max: float = tau_max_default,
        tau_sampling: str = tau_sampling_default,
        weight: float = 0.5
    ):
        self.N_tau = N_tau
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_sampling = tau_sampling
        self.weight = weight
    
    def analyze(self, time: np.ndarray, data: np.ndarray, **kwargs) -> Tuple:
        """Perform Elastic Net analysis."""
        from .me_analysis import perform_elastic_analysis
        
        return perform_elastic_analysis(
            time=time,
            data_baseline_subtracted=data,
            N_tau=kwargs.get('N_tau', self.N_tau),
            alpha=kwargs.get('alpha', self.alpha),
            tau_min=kwargs.get('tau_min', self.tau_min),
            tau_max=kwargs.get('tau_max', self.tau_max),
            tau_sampling=kwargs.get('tau_sampling', self.tau_sampling),
            weight=kwargs.get('weight', self.weight)
        )
    
    def get_name(self) -> str:
        return "Elastic Net (L1+L2)"


class FixedTauAnalysisStrategy(AnalysisStrategy):
    """
    Fixed relaxation times analysis strategy.
    """
    
    def __init__(
        self,
        approx_taus: tuple = (0.01, 0.1, 1.0),
        fit_baseline: bool = False
    ):
        self.approx_taus = approx_taus
        self.fit_baseline = fit_baseline
    
    def analyze(self, time: np.ndarray, data: np.ndarray, **kwargs) -> Tuple:
        """Perform fixed-tau analysis."""
        from .me_analysis import perform_three_exp_analysis
        
        params, fitted_data = perform_three_exp_analysis(
            time=time,
            data=data,
            approx_taus=kwargs.get('approx_taus', self.approx_taus),
            fit_baseline=kwargs.get('fit_baseline', self.fit_baseline)
        )
        
        # Return in consistent format: (tau, amplitudes, fitted_data, norms)
        tau = np.array(self.approx_taus)
        if self.fit_baseline and params.shape[1] == 4:
            amplitudes = params[:, 1:]  # Skip baseline column
        else:
            amplitudes = params
        
        return tau, amplitudes, fitted_data, None
    
    def get_name(self) -> str:
        return f"Fixed tau ({len(self.approx_taus)} exp)"


class AnalysisStrategyFactory:
    """
    Factory for creating analysis strategies.
    """
    
    _strategies = {
        'l2': L2AnalysisStrategy,
        'elastic': ElasticNetAnalysisStrategy,
        'elastic_net': ElasticNetAnalysisStrategy,
        'fixed_taus': FixedTauAnalysisStrategy,
        'fixed_taus_baseline': lambda **kwargs: FixedTauAnalysisStrategy(fit_baseline=True, **kwargs)
    }
    
    @classmethod
    def create(cls, method: str, **kwargs) -> AnalysisStrategy:
        """
        Create an analysis strategy.
        
        Parameters
        ----------
        method : str
            Method name ('l2', 'elastic', 'fixed_taus', etc.)
        **kwargs : dict
            Parameters for the strategy constructor
            
        Returns
        -------
        AnalysisStrategy
            Instantiated strategy
        """
        if method not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown method '{method}'. Available: {available}")
        
        strategy_class = cls._strategies[method]
        if callable(strategy_class) and not isinstance(strategy_class, type):
            # It's a lambda, call it directly
            return strategy_class(**kwargs)
        
        return strategy_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, strategy_class: type):
        """Register a new strategy."""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def available_methods(cls) -> list:
        """Return list of available method names."""
        return list(cls._strategies.keys())


# Convenience function for backward compatibility
def perform_analysis(
    method: str,
    time: np.ndarray,
    data: np.ndarray,
    **kwargs
) -> Tuple:
    """
    Perform multi-exponential analysis using the specified method.
    
    This is a convenience wrapper around the strategy pattern.
    
    Parameters
    ----------
    method : str
        Analysis method ('l2', 'elastic', 'fixed_taus', etc.)
    time : np.ndarray
        Time axis
    data : np.ndarray
        Data to analyze
    **kwargs : dict
        Method-specific parameters
        
    Returns
    -------
    tuple
        (tau, amplitudes, fitted_data, additional_info)
        
    Examples
    --------
    >>> tau, amps, fit, norms = perform_analysis(
    ...     'l2', time, data, N_tau=100, alpha=0.1
    ... )
    """
    strategy = AnalysisStrategyFactory.create(method, **kwargs)
    logging.info(f"Performing {strategy.get_name()} analysis...")
    return strategy.analyze(time, data, **kwargs)

