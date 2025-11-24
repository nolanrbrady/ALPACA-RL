"""State validation and categorical helpers for ALPACA."""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


class StateValidator:
    """Validate and sanitize patient states against bounds and categorical constraints."""

    def __init__(self, observation_cols: Sequence[str], variable_bounds: pd.DataFrame, y_categorical_groups: Dict[str, Iterable[str]]):
        if variable_bounds is None:
            raise ValueError("Variable bounds DataFrame must be provided.")
        self.observation_cols = list(observation_cols)
        self.variable_bounds = variable_bounds
        self.y_categorical_groups = y_categorical_groups or {}

    def enforce_categorical_groups(self, sample_series: pd.Series) -> pd.Series:
        """Project one-hot categorical groups onto a valid simplex (single 1.0)."""
        for _, cols in self.y_categorical_groups.items():
            cols = [c for c in cols if c in sample_series.index]
            if not cols:
                continue
            values = sample_series[cols].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                values = np.nan_to_num(values, nan=0.0)
            if values.sum() <= 0.0:
                winner_idx = int(np.argmax(values))
            else:
                winner_idx = int(np.argmax(values))
            for idx, col in enumerate(cols):
                sample_series[col] = 1.0 if idx == winner_idx else 0.0
        return sample_series

    def clip_sample_to_bounds(self, sample_series: pd.Series) -> pd.Series:
        """Clip sampled continuous features to ADNI variable bounds."""
        for col in sample_series.index:
            if col not in self.variable_bounds.index:
                continue
            lower_bound = float(self.variable_bounds.loc[col, 'lower_bound'])
            upper_bound = float(self.variable_bounds.loc[col, 'upper_bound'])
            clipped_val = np.clip(float(sample_series[col]), lower_bound, upper_bound)
            sample_series[col] = np.float32(clipped_val)
        return sample_series

    def check_state_bounds(self, state_values: np.ndarray) -> Tuple[bool, List[Dict[str, object]]]:
        """
        Check if state values are within the acceptable ADNI variable bounds.

        Args:
            state_values (numpy.ndarray): Array of state values corresponding to observation_cols

        Returns:
            tuple: (is_within_bounds: bool, out_of_bounds_variables: list)
        """
        state_series = pd.Series(state_values, index=self.observation_cols)
        out_of_bounds_vars = []

        for var_name, value in state_series.items():
            if var_name not in self.variable_bounds.index:
                continue

            lower_bound = float(self.variable_bounds.loc[var_name, 'lower_bound'])
            upper_bound = float(self.variable_bounds.loc[var_name, 'upper_bound'])
            tolerance = self._bound_tolerance(lower_bound, upper_bound)
            value_f = float(value)

            if not np.isfinite(value_f):
                out_of_bounds_vars.append(
                    {
                        'variable': var_name,
                        'value': value_f,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'violation': 'non_finite',
                    }
                )
                continue

            if value_f < lower_bound - tolerance or value_f > upper_bound + tolerance:
                violation_type = 'below_lower' if value_f < lower_bound else 'above_upper'
                out_of_bounds_vars.append(
                    {
                        'variable': var_name,
                        'value': value_f,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'violation': violation_type,
                    }
                )

        is_within_bounds = len(out_of_bounds_vars) == 0
        return is_within_bounds, out_of_bounds_vars

    def _bound_tolerance(self, lower_bound: float, upper_bound: float) -> float:
        """Compute an absolute tolerance for bound comparisons based on value magnitudes."""
        scale = max(abs(lower_bound), abs(upper_bound), 1.0)
        return max(1e-6, 1e-6 * scale)
