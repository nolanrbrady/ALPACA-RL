"""Action validation and constraint enforcement for ALPACA."""

from typing import Dict, Optional, Tuple

import numpy as np


class ActionValidator:
    """Validate actions and enforce mutual exclusivity constraints."""

    def __init__(self, action_cols: list[str], no_med_idx: int):
        self.action_cols = action_cols
        self.no_med_idx = no_med_idx

    def validate_action_structure(self, action: np.ndarray) -> None:
        """Validate action shape and binary values."""
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a numpy array.")
        if action.shape != (len(self.action_cols),):
            raise ValueError(
                f"Action must have shape ({len(self.action_cols)},); received shape {action.shape}."
            )
        if not np.isfinite(action).all():
            raise ValueError("Action contains non-finite values.")
        unique_vals = set(np.unique(action))
        if not unique_vals.issubset({0, 1}):
            raise ValueError("Action must be binary (0/1) for each dimension.")

    def check_constraints(self, action: np.ndarray, current_seq_len: int) -> Optional[Tuple[float, Dict[str, object]]]:
        """Enforce mutual exclusivity and mandate at least one active treatment."""
        # Rule 1: "No Medication" is exclusive
        if action[self.no_med_idx] == 1 and np.sum(action) > 1:
            info = {
                'constraint_violation': 'no_medication_with_other_actions',
                'sequence_length': current_seq_len + 1,
            }
            return -10.0, info
        
        # Rule 2: Must take at least one action
        if np.sum(action) == 0:
            info = {
                'constraint_violation': 'no_action_taken',
                'sequence_length': current_seq_len + 1,
            }
            return -10.0, info
            
        return None

