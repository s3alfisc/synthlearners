# synthlearners/simulator.py
import numpy as np
from scipy.special import expit
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
from abc import ABC, abstractmethod


class BaseDGP(ABC):
    """Abstract base class for data generating processes."""

    @abstractmethod
    def generate(self, N: int, T: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate panel data."""
        pass


class FactorDGP(BaseDGP):
    """Factor model DGP with time trends."""

    def __init__(
        self,
        K: int = 4,
        unit_fac_lb: float = -5,
        unit_fac_ub: float = 3,
        sigma: float = 0.1,
        trend_sigma: float = 0.01,
    ):
        self.K = K
        self.unit_fac_lb = unit_fac_lb
        self.unit_fac_ub = unit_fac_ub
        self.sigma = sigma
        self.trend_sigma = trend_sigma

    def generate(self, N: int, T: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate panel data from factor model.

        Args:
            N: Number of units
            T: Number of time periods
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of (Y, L) where Y is outcomes and L is unit factors
        """
        F = np.random.rand(T, self.K)
        L = np.random.uniform(self.unit_fac_lb, self.unit_fac_ub, (N, self.K))
        time_trends = np.random.normal(0, self.trend_sigma, (N, 1)) * np.arange(
            T
        ).reshape(1, T)
        epsilon = np.random.normal(0, self.sigma, (N, T))
        Y = np.dot(L, F.T) + epsilon + time_trends
        return Y, L


@dataclass
class SimulationConfig:
    """Configuration for panel data simulation."""

    N: int = 100
    T: int = 50
    T_pre: int = 40
    n_treated: int = 1
    selection_mean: float = 1.0
    dgp: BaseDGP = FactorDGP()
    random_seed: Optional[int] = None
    treatment_effect: Union[float, np.ndarray] = 0.0  # New parameter


class PanelSimulator:
    """Simulator for synthetic control panel data."""

    def __init__(self, config: SimulationConfig):
        """Initialize simulator with configuration."""
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Validate and process treatment effect
        self._process_treatment_effect()

    def _process_treatment_effect(self):
        """Process and validate treatment effect specification."""
        effect = self.config.treatment_effect
        T_post = self.config.T - self.config.T_pre

        if isinstance(effect, (int, float)):
            # Constant treatment effect
            self.treatment_effect = np.full((self.config.n_treated, T_post), effect)
        elif isinstance(effect, np.ndarray):
            # Heterogeneous treatment effects
            if effect.shape != (self.config.n_treated, T_post):
                raise ValueError(
                    f"Treatment effect array must have shape ({self.config.n_treated}, {T_post}), "
                    f"got {effect.shape}"
                )
            self.treatment_effect = effect
        else:
            raise TypeError(
                "Treatment effect must be a number or numpy array, "
                f"got {type(effect)}"
            )

    def assign_treatment(self, L: np.ndarray) -> np.ndarray:
        """Assign treatment based on unit factors.

        Args:
            L: Unit factors matrix

        Returns:
            Array of treated unit indices
        """
        sel_coef = np.random.normal(self.config.selection_mean, 1, L.shape[1])
        pscore = expit(L @ sel_coef**2)
        pscore /= pscore.sum()
        return np.random.choice(
            range(L.shape[0]), self.config.n_treated, replace=False, p=pscore
        )

    def apply_treatment_effect(
        self, Y: np.ndarray, treated_units: np.ndarray
    ) -> np.ndarray:
        """Apply treatment effects to the outcome matrix.

        Args:
            Y: Original outcome matrix
            treated_units: Indices of treated units

        Returns:
            Modified outcome matrix with treatment effects
        """
        Y_modified = Y.copy()
        T_pre = self.config.T_pre

        # Apply treatment effects to each treated unit
        for i, unit_idx in enumerate(treated_units):
            Y_modified[unit_idx, T_pre:] += self.treatment_effect[i]

        return Y_modified

    def simulate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate one simulation of panel data.

        Returns:
            Tuple of (Y, Y_cf, L, treated_units) where:
                Y: Panel data array with treatment effects
                Y_cf: Counterfactual panel data array without treatment
                L: Unit factors array
                treated_units: Array of treated unit indices
        """
        Y_cf, L = self.config.dgp.generate(self.config.N, self.config.T)
        treated_units = self.assign_treatment(L)

        # Apply treatment effects
        Y = self.apply_treatment_effect(Y_cf, treated_units)

        return Y, Y_cf, L, treated_units

    def simulate_batch(
        self, n_sims: int, parallel: bool = False, n_jobs: int = -1
    ) -> list:
        """Generate multiple simulations.

        Args:
            n_sims: Number of simulations to generate
            parallel: Whether to run simulations in parallel
            n_jobs: Number of parallel jobs (if parallel=True)

        Returns:
            List of (Y, Y_cf, L, treated_units) tuples
        """
        if not parallel:
            return [self.simulate() for _ in range(n_sims)]

        from joblib import Parallel, delayed

        return Parallel(n_jobs=n_jobs)(delayed(self.simulate)() for _ in range(n_sims))
