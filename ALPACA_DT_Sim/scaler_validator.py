"""Validation helpers for ALPACA scaler artifacts."""

from typing import Dict, List, Sequence

import numpy as np

from .env_constants import DELTA_COLUMN


class ScalerValidator:
    """Validate and align scaler metadata against schema expectations."""

    def __init__(
        self,
        model_input_cols: Sequence[str],
        cont_obs_cols: Sequence[str],
        model_cont_output_cols: Sequence[str],
    ):
        self.model_input_cols = list(model_input_cols)
        self.cont_obs_cols = list(cont_obs_cols)
        self.model_cont_output_cols = list(model_cont_output_cols)

    def align_scaler_X_feature_names(self, scaler_X) -> None:
        """Align scaler_X feature names to match schema/model input expectations.

        - Map any time-gap aliases (e.g., 'time_delta', 'time_since_prev') to the
          configured delta column.
        - Drop absolute time like 'months_since_bl' which is not part of model inputs.
        - Remove any names not present in model_input_cols.
        - Avoid duplicates and preserve the first occurrence stats.
        """
        if scaler_X is None or not hasattr(scaler_X, 'feature_names_in_'):
            return
        names = list(scaler_X.feature_names_in_)
        means = np.array(getattr(scaler_X, 'mean_', []), dtype=float)
        scales = np.array(getattr(scaler_X, 'scale_', []), dtype=float) if hasattr(scaler_X, 'scale_') else None
        vars_ = np.array(getattr(scaler_X, 'var_', []), dtype=float) if hasattr(scaler_X, 'var_') else None

        new_names: list[str] = []
        new_means: list[float] = []
        new_scales: list[float] = []
        new_vars: list[float] = []

        def maybe_append(target_name: str, i: int):
            if target_name in new_names:
                return
            if target_name not in self.model_input_cols:
                return
            new_names.append(target_name)
            if i < len(means):
                new_means.append(float(means[i]))
            if scales is not None and i < len(scales):
                new_scales.append(float(scales[i]))
            if vars_ is not None and i < len(vars_):
                new_vars.append(float(vars_[i]))

        for i, name in enumerate(names):
            if name in ('months_since_bl',):
                continue
            if name in ('time_delta', 'time_since_prev'):
                maybe_append(DELTA_COLUMN, i)
            else:
                maybe_append(name, i)

        scaler_X.feature_names_in_ = np.array(new_names, dtype=object)
        scaler_X.n_features_in_ = len(new_names)
        if hasattr(scaler_X, 'mean_'):
            scaler_X.mean_ = np.array(new_means, dtype=float)
        if hasattr(scaler_X, 'scale_') and len(new_scales) > 0:
            scaler_X.scale_ = np.array(new_scales, dtype=float)
        if hasattr(scaler_X, 'var_') and len(new_vars) > 0:
            scaler_X.var_ = np.array(new_vars, dtype=float)

    def validate_alignment(self, scaler_X, scaler_y) -> None:
        """Fail fast if scaler metadata does not align with schema/model expectations."""
        if scaler_X is None or scaler_y is None:
            raise ValueError("Both scaler_X and scaler_y must be initialized before validation.")

        sx_names = list(getattr(scaler_X, 'feature_names_in_', []))
        if not sx_names:
            raise ValueError("scaler_X is missing feature_names_in_; regenerate preprocessing artifacts.")
        self._raise_if_duplicates(sx_names, "scaler_X")
        self._validate_scaler_stats_length(scaler_X, 'scaler_X')

        extra_inputs = [c for c in sx_names if c not in self.model_input_cols]
        if extra_inputs:
            raise ValueError(
                "scaler_X contains features not present in model_input_cols: "
                f"{extra_inputs}. Regenerate artifacts to realign training and environment."
            )
        expected_scaled_inputs = [c for c in self.cont_obs_cols + [DELTA_COLUMN] if c in self.model_input_cols]
        missing_inputs = [c for c in expected_scaled_inputs if c not in sx_names]
        if missing_inputs:
            raise ValueError(
                "scaler_X is missing required continuous inputs expected by the model: "
                f"{missing_inputs}. Regenerate preprocessing artifacts."
            )
        if DELTA_COLUMN not in sx_names:
            raise ValueError(f"scaler_X must include the time-delta column '{DELTA_COLUMN}'.")

        sy_names = list(getattr(scaler_y, 'feature_names_in_', []))
        if not sy_names:
            raise ValueError("scaler_y is missing feature_names_in_; regenerate preprocessing artifacts.")
        self._raise_if_duplicates(sy_names, "scaler_y")
        self._validate_scaler_stats_length(scaler_y, 'scaler_y')

        missing_outputs = [c for c in self.model_cont_output_cols if c not in sy_names]
        if missing_outputs:
            raise ValueError(
                "scaler_y is missing required model continuous outputs: "
                f"{missing_outputs}. Regenerate preprocessing artifacts."
            )

    def _raise_if_duplicates(self, columns: List[str], scaler_name: str) -> None:
        duplicates = self._find_duplicate_columns(columns)
        if duplicates:
            raise ValueError(f"{scaler_name} contains duplicate feature names: {duplicates}")

    def _find_duplicate_columns(self, columns: List[str]) -> List[str]:
        seen = set()
        duplicates = []
        for col in columns:
            if col in seen and col not in duplicates:
                duplicates.append(col)
            seen.add(col)
        return duplicates

    def _validate_scaler_stats_length(self, scaler, scaler_name: str) -> None:
        """Ensure scaler statistics arrays match feature_names_in_ length."""
        n_features = len(getattr(scaler, 'feature_names_in_', []))
        for attr in ('mean_', 'scale_', 'var_'):
            if hasattr(scaler, attr):
                values = getattr(scaler, attr)
                if values is None:
                    continue
                if len(values) != n_features:
                    raise ValueError(
                        f"{scaler_name} attribute '{attr}' length {len(values)} "
                        f"does not match number of features {n_features}."
                    )
