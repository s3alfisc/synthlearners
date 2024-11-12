from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
from enum import Enum

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.stats import norm
import matplotlib.pyplot as plt
import pyensmallen as pe

from joblib import Parallel, delayed


######################################################################


class SynthMethod(Enum):
    """Enumeration of available synthetic control methods."""

    LP_NORM = "lp_norm"
    LINEAR = "linear"
    SIMPLEX = "simplex"


@dataclass
class SynthResults:
    """Container for synthetic control results."""

    weights: np.ndarray
    treated_outcome: np.ndarray
    synthetic_outcome: np.ndarray
    pre_treatment_rmse: float
    post_treatment_effect: float
    method: SynthMethod
    p: Optional[float] = None
    jackknife_effects: Optional[np.ndarray] = None

    def treatment_effect(self) -> np.ndarray:
        """Calculate treatment effect."""
        return self.treated_outcome - self.synthetic_outcome

    def att(self) -> float:
        """Calculate average treatment effect on the treated."""
        return self.post_treatment_effect

    def jackknife_variance(self) -> Optional[np.ndarray]:
        """Calculate jackknife variance of treatment effects."""
        if self.jackknife_effects is None:
            return None
        return np.var(self.jackknife_effects, axis=0)

    def confidence_intervals(
        self, alpha: float = 0.05
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals using jackknife variance.

        Args:
            alpha: Significance level (default: 0.05 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound) arrays or None if jackknife not available
        """
        if self.jackknife_effects is None:
            return None

        effects = self.treatment_effect()
        std_err = np.sqrt(self.jackknife_variance())
        z_score = norm.ppf(1 - alpha / 2)

        lower = effects - z_score * std_err
        upper = effects + z_score * std_err

        return lower, upper


class Synth:
    """Base class for synthetic control estimators."""

    def __init__(
        self,
        method: Union[str, SynthMethod] = "simplex",
        p: float = 1.0,
        max_iterations: int = 10000,
        tolerance: float = 1e-8,
        n_jobs: int = 4,
    ):
        """Initialize synthetic control estimator.

        Args:
            method: Estimation method ('lp_norm', 'linear', or 'simplex')
            p: L-p norm constraint (only used if method='lp_norm')
            max_iterations: Maximum number of iterations for optimization
            tolerance: Convergence tolerance for optimization
        """
        self.method = SynthMethod(method) if isinstance(method, str) else method
        self.p = p
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights_ = None
        self.n_jobs = n_jobs

    def fit(
        self,
        Y: np.ndarray,
        treated_units: Union[int, np.ndarray],
        T_pre: int,
        T_post: Optional[int] = None,
        compute_jackknife: bool = True,
    ) -> SynthResults:
        """Fit synthetic control model."""
        treated_units = (
            np.array([treated_units])
            if isinstance(treated_units, int)
            else treated_units
        )
        T_post = Y.shape[1] - T_pre if T_post is None else T_post
        self.T_pre = T_pre

        # Split data into treated and control groups
        Y_treated = Y[treated_units, :].mean(axis=0)
        control_units = np.setdiff1d(range(Y.shape[0]), treated_units)
        Y_control = Y[control_units, :]

        # Solve for weights using specified method
        if self.method == SynthMethod.LP_NORM:
            self.weights_ = self._solve_lp_norm(Y_control[:, :T_pre], Y_treated[:T_pre])
        elif self.method == SynthMethod.LINEAR:
            self.weights_ = self._solve_linear(Y_control[:, :T_pre], Y_treated[:T_pre])
        else:  # SIMPLEX
            self.weights_ = self._solve_simplex(Y_control[:, :T_pre], Y_treated[:T_pre])

        # Compute synthetic control unit
        synthetic_outcome = np.dot(Y_control.T, self.weights_)

        # Calculate pre-treatment fit
        pre_rmse = np.sqrt(
            np.mean((Y_treated[:T_pre] - synthetic_outcome[:T_pre]) ** 2)
        )

        # Compute jackknife effects if requested and possible
        jackknife_effects = None
        if compute_jackknife:
            jackknife_effects = self._compute_jackknife_effects(Y, treated_units, T_pre)

        return SynthResults(
            weights=self.weights_,
            treated_outcome=Y_treated,
            synthetic_outcome=synthetic_outcome,
            post_treatment_effect=np.mean(
                Y_treated[T_pre : T_pre + T_post]
                - synthetic_outcome[T_pre : T_pre + T_post]
            ),
            pre_treatment_rmse=pre_rmse,
            method=self.method,
            p=self.p if self.method == SynthMethod.LP_NORM else None,
            jackknife_effects=jackknife_effects,
        )

    def plot(
        self,
        results: SynthResults,
        Y: np.ndarray,
        treated_units: np.ndarray,
        T_pre: int,
        mode: Literal["raw", "effect"] = "raw",
        show_ci: bool = True,
        alpha: float = 0.05,
        figsize: Tuple[int, int] = (10, 6),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot synthetic control results.

        Args:
            results: SynthResults object
            Y: Original panel data
            treated_units: Treated unit indices
            T_pre: Pre-treatment period
            mode: 'raw' for original data or 'effect' for treatment effects
            show_ci: Whether to show confidence intervals (only for 'effect' mode)
            alpha: Significance level for confidence intervals
            figsize: Figure size
            ax: Optional matplotlib Axes object

        Returns:
            matplotlib Axes object
        """
        if ax is None:
            f, ax = plt.subplots(figsize=figsize)

        if mode == "raw":
            # Plot raw data
            ctrl_units = np.setdiff1d(range(Y.shape[0]), treated_units)

            # Plot control units in background
            ax.plot(Y[ctrl_units].T, color="gray", alpha=0.2, linestyle="--")

            # Plot treated and synthetic trajectories
            ax.plot(results.treated_outcome, label="Treated", color="blue", linewidth=2)
            ax.plot(
                results.synthetic_outcome,
                label="Synthetic Control",
                color="red",
                linewidth=2,
                linestyle="--",
            )

            # Add treatment line
            ax.axvline(T_pre, color="black", linestyle="--", label="Treatment")

            ax.set_title("Raw Trajectories")
            ax.legend()

        elif mode == "effect":
            # Compute and plot treatment effects
            effects = results.treatment_effect()

            # Center x-axis at treatment time
            t = np.arange(len(effects)) - T_pre

            ax.plot(t, effects, color="blue", linewidth=2, label="Treatment Effect")
            ax.axvline(0, color="black", linestyle="--", label="Treatment")
            ax.axhline(0, color="gray", linestyle=":")

            if show_ci and results.jackknife_effects is not None:
                ci = results.confidence_intervals(alpha)
                if ci is not None:
                    lower, upper = ci
                    ax.fill_between(
                        t,
                        lower,
                        upper,
                        alpha=0.2,
                        color="blue",
                        label=f"{int((1-alpha)*100)}% CI",
                    )

            ax.set_title(f"Treatment Effect \n ATT: {results.att():.2f}")
            ax.set_xlabel("Time Relative to Treatment")
            ax.set_ylabel("Effect Size")
            ax.legend()

        return ax

    def _objective(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Compute objective function (squared error loss)."""
        return np.sum((np.dot(X, w) - y) ** 2)

    def _gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of objective function."""
        return 2 * np.dot(X.T, (np.dot(X, w) - y))

    def _solve_lp_norm(
        self, Y_control: np.ndarray, Y_treated: np.ndarray
    ) -> np.ndarray:
        """Solve synthetic control problem using Frank-Wolfe with Lp norm constraint."""
        N_control = Y_control.shape[0]

        def f(w, grad):
            if grad.size > 0:
                grad[:] = self._gradient(w, Y_control.T, Y_treated)
            return self._objective(w, Y_control.T, Y_treated)

        optimizer = pe.FrankWolfe(
            p=self.p, max_iterations=self.max_iterations, tolerance=self.tolerance
        )
        initial_w = np.ones(N_control) / N_control
        return optimizer.optimize(f, initial_w)

    def _solve_linear(self, Y_control: np.ndarray, Y_treated: np.ndarray) -> np.ndarray:
        """Solve synthetic control problem using ordinary least squares."""
        return np.linalg.lstsq(Y_control.T, Y_treated, rcond=None)[0]

    def _solve_simplex(
        self, Y_control: np.ndarray, Y_treated: np.ndarray
    ) -> np.ndarray:
        """Solve synthetic control problem with simplex constraints."""
        N_control = Y_control.shape[0]
        initial_w = np.repeat(1 / N_control, N_control)
        bounds = tuple((0, 1) for _ in range(N_control))

        weights = fmin_slsqp(
            func=lambda w: self._objective(w, Y_control.T, Y_treated),
            x0=initial_w,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=bounds,
            disp=False,
        )
        return weights

    def _jackknife_single_run(
        self, Y: np.ndarray, treated_units: np.ndarray, T_pre: int, leave_out_idx: int
    ) -> np.ndarray:
        """Compute single jackknife iteration.

        Args:
            Y: Panel data array
            treated_units: Array of treated unit indices
            T_pre: Pre-treatment period cutoff
            leave_out_idx: Index of treated unit to leave out

        Returns:
            Treatment effect for this jackknife iteration
        """
        # Create a new instance with same parameters for thread safety
        synth_instance = Synth(
            method=self.method,
            p=self.p,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        # Leave out one treated unit
        jackknife_treated = np.delete(treated_units, leave_out_idx)

        # Fit synthetic control on reduced sample
        results = synth_instance.fit(
            Y, jackknife_treated, T_pre, compute_jackknife=False
        )
        return results.treatment_effect()

    def _compute_jackknife_effects(
        self,
        Y: np.ndarray,
        treated_units: np.ndarray,
        T_pre: int,
    ) -> Optional[np.ndarray]:
        """Compute jackknife treatment effects in parallel.

        Args:
            Y: Panel data array
            treated_units: Array of treated unit indices
            T_pre: Pre-treatment period cutoff

        Returns:
            Array of jackknife treatment effects or None if n_treated <= 1
        """
        if len(treated_units) <= 1:
            return None

        n_treated = len(treated_units)

        # Use parallel processing with default batch size
        effects = Parallel(n_jobs=self.n_jobs)(
            delayed(self._jackknife_single_run)(Y, treated_units, T_pre, i)
            for i in range(n_treated)
        )

        return np.array(effects)
