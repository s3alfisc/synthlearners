from dataclasses import dataclass
from typing import Optional, Union, Tuple, Literal
from enum import Enum

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .utils import tqdm_joblib
from .solvers import _solve_lp_norm, _solve_linear, _solve_simplex, _solve_matching


######################################################################


class SynthMethod(Enum):
    """Enumeration of available synthetic control methods."""

    LP_NORM = "lp_norm"
    LINEAR = "linear"
    SIMPLEX = "simplex"
    MATCHING = "matching"


@dataclass
class SynthResults:
    """Container for synthetic control results."""

    def __init__(
        self,
        unit_weights: np.ndarray,
        treated_outcome: np.ndarray,
        synthetic_outcome: np.ndarray,
        pre_treatment_rmse: float,
        post_treatment_effect: float,
        method: "SynthMethod",
        p: Optional[float] = None,
        jackknife_effects: Optional[np.ndarray] = None,
        permutation_p_value: Optional[float] = None,
    ):
        self.unit_weights = unit_weights
        self.treated_outcome = treated_outcome
        self.synthetic_outcome = synthetic_outcome
        self.pre_treatment_rmse = pre_treatment_rmse
        self.post_treatment_effect = post_treatment_effect
        self.method = method
        self.p = p
        self.jackknife_effects = jackknife_effects
        self.permutation_p_value = permutation_p_value

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
        """Calculate confidence intervals using jackknife variance."""
        if self.jackknife_effects is None:
            return None

        effects = self.treatment_effect()
        std_err = np.sqrt(self.jackknife_variance())
        z_score = norm.ppf(1 - alpha / 2)

        lower = effects - z_score * std_err
        upper = effects + z_score * std_err

        return lower, upper


class Synth:
    """Base class for synthetic control estimators with support for individual unit matching."""

    def __init__(
        self,
        method: Union[str, SynthMethod] = "simplex",
        p: float = 1.0,
        intercept: bool = False,
        weight_type: str = "unit",
        max_iterations: int = 10000,
        tolerance: float = 1e-8,
        n_jobs: int = 8,
        granular_weights: bool = False,
    ):
        """Initialize synthetic control estimator.

        Args:
            method: Estimation method ('lp_norm', 'linear', 'simplex', or 'matching')
            p: L-p norm constraint (only used if method='lp_norm')
            intercept: Whether to include an intercept term
            weight_type: Type of weights to use ('unit', 'time', or 'both')
            max_iterations: Maximum number of iterations for optimization
            tolerance: Convergence tolerance for optimization
            n_jobs: Number of parallel jobs for jackknife
            granular_weights: Whether to compute unit-specific weights for each treated unit
        """
        self.method = SynthMethod(method) if isinstance(method, str) else method
        self.p = p
        self.intercept = intercept
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weight_type = weight_type
        self.n_jobs = n_jobs
        self.granular_weights = granular_weights
        self.unit_weights = None
        self.time_weights = None

    def fit(
        self,
        Y: np.ndarray,
        treated_units: Union[int, np.ndarray],
        T_pre: int,
        T_post: Optional[int] = None,
        compute_jackknife: bool = False,
        compute_permutation: bool = False,
        **kwargs,
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
        control_units = np.setdiff1d(range(Y.shape[0]), treated_units)
        Y_control = Y[control_units, :]

        if self.intercept:
            Y_control2 = np.r_[Y_control, np.ones((1, Y_control.shape[1]))]
        else:
            Y_control2 = Y_control

        if self.weight_type != "unit":
            raise NotImplementedError("Only 'unit' weights are currently supported.")

        Y_ctrl_pre = Y_control2[:, :T_pre]

        if self.granular_weights:
            # Compute weights and synthetic outcomes for each treated unit
            individual_weights = []
            individual_synthetic = []

            for treated_idx in treated_units:
                Y_treat_pre = Y[treated_idx, :T_pre]

                # Get weights for this treated unit
                if self.method == SynthMethod.LP_NORM:
                    weights = _solve_lp_norm(
                        Y_ctrl_pre,
                        Y_treat_pre,
                        self.p,
                        self.max_iterations,
                        self.tolerance,
                        **kwargs,
                    )
                elif self.method == SynthMethod.LINEAR:
                    weights = _solve_linear(Y_ctrl_pre, Y_treat_pre, **kwargs)
                elif self.method == SynthMethod.MATCHING:
                    weights = _solve_matching(
                        Y_ctrl_pre,
                        Y_treat_pre,
                        **kwargs,
                    )
                elif self.method == SynthMethod.SIMPLEX:
                    weights = _solve_simplex(
                        Y_ctrl_pre,
                        Y_treat_pre,
                        **kwargs,
                    )
                else:
                    raise NotImplementedError(
                        f"Method {self.method} not implemented. Please select one of ['lp_norm', 'linear', 'simplex', 'matching']"
                    )

                individual_weights.append(weights)
                synthetic = np.dot(Y_control2.T, weights)
                individual_synthetic.append(synthetic)

            # Average the weights and synthetic outcomes
            self.unit_weights = np.mean(individual_weights, axis=0)
            synthetic_outcome = np.mean(individual_synthetic, axis=0)
            Y_treated = Y[treated_units].mean(
                axis=0
            )  # Still need average treated outcome

        else:
            # Original behavior: average treated units first, then find weights
            # Y_treated = Y[treated_units].mean(axis=0)
            Y_treated = Y[treated_units].reshape(-1, Y.shape[1]).mean(axis=0)
            Y_treat_pre = Y_treated[:T_pre]

            if self.method == SynthMethod.LP_NORM:
                self.unit_weights = _solve_lp_norm(
                    Y_ctrl_pre,
                    Y_treat_pre,
                    self.p,
                    self.max_iterations,
                    self.tolerance,
                    **kwargs,
                )
            elif self.method == SynthMethod.LINEAR:
                self.unit_weights = _solve_linear(Y_ctrl_pre, Y_treat_pre, **kwargs)
            elif self.method == SynthMethod.MATCHING:
                self.unit_weights = _solve_matching(Y_ctrl_pre, Y_treat_pre, **kwargs)
            elif self.method == SynthMethod.SIMPLEX:
                self.unit_weights = _solve_simplex(Y_ctrl_pre, Y_treat_pre, **kwargs)
            else:
                raise NotImplementedError(
                    f"Method {self.method} not implemented. Please select one of ['lp_norm', 'linear', 'simplex', 'matching']"
                )

            synthetic_outcome = np.dot(Y_control2.T, self.unit_weights)

        # Calculate pre-treatment fit
        pre_rmse = np.sqrt(
            np.mean((Y_treated[:T_pre] - synthetic_outcome[:T_pre]) ** 2)
        )

        # Compute jackknife effects if requested and possible
        jackknife_effects = None
        if compute_jackknife:
            jackknife_effects = self._compute_jackknife_effects(Y, treated_units, T_pre)

        # Calculate permutation p-value if requested
        permutation_p_value = None
        if compute_permutation:
            permutation_p_value = self._compute_permutation_p_value(
                Y, treated_units, T_pre
            )

        return SynthResults(
            unit_weights=self.unit_weights,
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
            permutation_p_value=permutation_p_value,
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

            if results.permutation_p_value is not None:
                ts = f"Treatment Effect \n ATT: {results.att():.2f} (p={results.permutation_p_value:.3f})"
            else:
                ts = f"Treatment Effect \n ATT: {results.att():.2f}"
            ax.set_title(ts)
            ax.set_xlabel("Time Relative to Treatment")
            ax.set_ylabel("Effect Size")
            ax.legend()

        return ax

    def _jackknife_single_run(
        self, Y: np.ndarray, treated_units: np.ndarray, T_pre: int, leave_out_idx: int
    ) -> np.ndarray:
        """Run single jackknife iteration leaving out one unit."""
        # Create a copy of Y without the left-out unit
        Y_reduced = np.delete(Y, leave_out_idx, axis=0)

        # Adjust treated_units indices to account for removal
        adjusted_treated = np.array(
            [
                i if i < leave_out_idx else i - 1
                for i in treated_units
                if i != leave_out_idx
            ]
        )

        # Create new instance with same parameters
        synth_instance = Synth(
            method=self.method,
            p=self.p,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        # Fit on reduced sample
        results = synth_instance.fit(
            Y_reduced,
            adjusted_treated,
            T_pre,
            compute_jackknife=False,
            compute_permutation=False,
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

        n = Y.shape[0]

        # Create progress bar and run parallel computation
        with tqdm_joblib(
            tqdm(total=n, desc="Computing jackknife estimates")
        ) as progress_bar:
            effects = Parallel(n_jobs=self.n_jobs)(
                delayed(self._jackknife_single_run)(Y, treated_units, T_pre, i)
                for i in range(n)
            )

        return np.array(effects)

    def _compute_permutation_p_value(
        self, Y: np.ndarray, treated_units: np.ndarray, T_pre: int
    ) -> float:
        """Compute permutation test p-value."""
        # Get the true effect for comparison
        true_results = self.fit(
            Y, treated_units, T_pre, compute_jackknife=False, compute_permutation=False
        )
        true_effect = np.abs(true_results.post_treatment_effect)

        n = Y.shape[0]
        if (n - 1) <= 20:
            print(
                f"You have {n} units, so the lowest possible p-value is {1/(n-1)}, which is smaller than traditional α of 0.05 \nPermutation test may be unreliable"
            )

        # Get control units
        control_units = np.setdiff1d(range(Y.shape[0]), treated_units)

        # Run parallel computation with progress bar
        with tqdm_joblib(
            tqdm(total=len(control_units), desc="Computing permutation test")
        ) as progress_bar:
            placebo_effects = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_placebo_effect)(
                    Y, control_unit, treated_units, T_pre
                )
                for control_unit in control_units
            )

        # Convert to absolute values for two-sided test
        placebo_effects = np.abs(placebo_effects)

        # Compute p-value as proportion of placebo effects larger than true effect
        p_value = np.mean(placebo_effects >= true_effect)

        return p_value

    def _compute_placebo_effect(
        self, Y: np.ndarray, placebo_unit: int, original_treated: np.ndarray, T_pre: int
    ) -> float:
        """Compute effect for a single placebo treatment."""
        # Remove original treated units from the data
        Y_reduced = np.delete(Y, original_treated, axis=0)

        # Adjust placebo unit index to account for removed treated units
        adjusted_placebo = placebo_unit - np.sum(original_treated < placebo_unit)

        # Create new instance with same parameters
        synth_instance = Synth(
            method=self.method,
            p=self.p,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )

        # Fit synthetic control using placebo unit as treated
        results = synth_instance.fit(
            Y_reduced,
            adjusted_placebo,
            T_pre,
            compute_jackknife=False,
            compute_permutation=False,
        )

        return results.post_treatment_effect


######################################################################
