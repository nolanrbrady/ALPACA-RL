import numpy as np
import pandas as pd


def manage_state_scaling(state_data: pd.DataFrame, scaler, normalize: bool = True) -> pd.DataFrame:
    """
    Scale or unscale only the intersection of columns present in both the DataFrame and scaler.

    Args:
        state_data: DataFrame whose columns correspond to feature names known by the scaler.
        scaler: A fitted sklearn-style scaler that exposes feature statistics.
        normalize: When True apply (x - mean) / scale, otherwise apply the inverse transform.

    Returns:
        pandas.DataFrame with transformed values for the overlapping columns.

    Raises:
        TypeError: If state_data is not a pandas DataFrame.
        ValueError: If a requested column is missing from the scaler metadata.
    """
    if not isinstance(state_data, pd.DataFrame):
        raise TypeError("Input 'state_data' must be a pandas DataFrame.")
    data = state_data.copy()
    if not hasattr(scaler, 'feature_names_in_'):
        return data

    name_to_idx = {c: i for i, c in enumerate(scaler.feature_names_in_)}
    with_mean = bool(getattr(scaler, 'with_mean', True))
    with_std = bool(getattr(scaler, 'with_std', True))

    for col in list(data.columns):
        if col not in name_to_idx:
            raise ValueError(f"Column {col} not found in scaler feature names")

        j = name_to_idx[col]
        col_vals = data[col].astype(float)
        if normalize:
            if with_mean:
                col_vals = col_vals - float(scaler.mean_[j])
            if with_std:
                scale_j = float(scaler.scale_[j]) if getattr(scaler, 'scale_', None) is not None else 1.0
                if scale_j != 0.0:
                    col_vals = col_vals / scale_j
        else:
            if with_std:
                scale_j = float(scaler.scale_[j]) if getattr(scaler, 'scale_', None) is not None else 1.0
                col_vals = col_vals * scale_j
            if with_mean:
                col_vals = col_vals + float(scaler.mean_[j])
        data[col] = col_vals.astype(np.float32)
    return data
