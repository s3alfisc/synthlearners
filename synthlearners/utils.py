import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


def prepare_panel(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    pre_treatment_period: Optional[Union[int, float]] = None,
) -> dict:
    """Enhanced version that returns additional information about the panel structure.

    Returns dictionary with keys:
        - Y: Outcome matrix (N x T)
        - treated_units: Array of treated unit indices
        - T_pre: Number of pre-treatment periods
        - unit_labels: Series mapping matrix rows to original unit IDs
        - time_periods: Array of original time period values
        - treatment_matrix: Binary treatment indicator matrix
    """
    # Basic data preparation
    Y, treated_units, T_pre = prepare_panel_data(
        df, unit_col, time_col, outcome_col, treatment_col, pre_treatment_period
    )

    # Get additional information
    unit_labels = pd.Series(df[unit_col].unique(), name="unit_labels")

    time_periods = np.sort(df[time_col].unique())

    treatment_matrix = df.pivot(
        index=unit_col, columns=time_col, values=treatment_col
    ).values

    return {
        "Y": Y,
        "W": treatment_matrix,
        "treated_units": treated_units,
        "T_pre": T_pre,
        "unit_labels": unit_labels,
        "time_periods": time_periods,
    }


def prepare_panel_data(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treatment_col: str,
    pre_treatment_period: Optional[Union[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert long panel data to format required by Synth.

    Args:
        df: Panel DataFrame in long format
        unit_col: Name of unit ID column
        time_col: Name of time period column
        outcome_col: Name of outcome variable column
        treatment_col: Name of treatment indicator column
        pre_treatment_period: Optional time period denoting treatment start.
            If None, inferred as first period where treatment == 1

    Returns:
        Tuple of (Y, treated_units, T_pre) where:
            Y: Outcome matrix (N x T)
            treated_units: Array of treated unit indices
            T_pre: Number of pre-treatment periods

    Example:
        >>> df = pd.DataFrame({
        ...     'outcome': [1, 2, 3, 4, 5, 6],
        ...     'treatment': [0, 1, 0, 0, 0, 1],
        ...     'unit': [1, 1, 2, 2, 3, 3],
        ...     'time': [1, 2, 1, 2, 1, 2]
        ... })
        >>> Y, treated, T_pre = prepare_panel_data(
        ...     df, 'unit', 'time', 'outcome', 'treatment'
        ... )
    """
    # Check if panel is balanced
    units_per_time = df.groupby(time_col)[unit_col].count()
    if not (units_per_time == units_per_time.iloc[0]).all():
        raise ValueError("Panel must be balanced (same units in all time periods)")

    # Create outcome matrix
    Y = df.pivot(index=unit_col, columns=time_col, values=outcome_col).values

    # Get treatment matrix
    W = df.pivot(index=unit_col, columns=time_col, values=treatment_col).values

    # Identify treated units (any unit that receives treatment)
    treated_units = np.where(W.sum(axis=1) > 0)[0]

    # Determine pre-treatment period
    if pre_treatment_period is None:
        # Find first period where any unit is treated
        first_treated_period = df[df[treatment_col] == 1][time_col].min()
        # Convert to array index (assuming time periods are sorted)
        unique_times = sorted(df[time_col].unique())
        T_pre = unique_times.index(first_treated_period)
    else:
        # Find index of pre_treatment_period in sorted unique times
        unique_times = sorted(df[time_col].unique())
        T_pre = unique_times.index(pre_treatment_period)

    return Y, treated_units, T_pre
