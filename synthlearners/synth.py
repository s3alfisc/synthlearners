import numpy as np
import pyensmallen as pe
from scipy.optimize import fmin_slsqp
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from enum import Enum

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
    method: SynthMethod
    p: Optional[float] = None

    def treatment_effect(self) -> np.ndarray:
        """Calculate treatment effect as difference between treated and synthetic outcomes."""
        return self.treated_outcome - self.synthetic_outcome


class Synth:
    """Base class for synthetic control estimators."""

    def __init__(
        self,
        method: Union[str, SynthMethod] = "simplex",
        p: float = 1.0,
        max_iterations: int = 10000,
        tolerance: float = 1e-8,
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

    def fit(
        self,
        Y: np.ndarray,
        treated_units: Union[int, np.ndarray],
        T_pre: int,
        T_post: Optional[int] = None,
    ) -> SynthResults:
        """Fit synthetic control model."""
        treated_units = (
            np.array([treated_units])
            if isinstance(treated_units, int)
            else treated_units
        )
        T_post = Y.shape[1] - T_pre if T_post is None else T_post

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

        return SynthResults(
            weights=self.weights_,
            treated_outcome=Y_treated,
            synthetic_outcome=synthetic_outcome,
            pre_treatment_rmse=pre_rmse,
            method=self.method,
            p=(
                self.p if self.method == SynthMethod.LP_NORM else None
            ),  # Pass p only for LP_NORM method
        )

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
