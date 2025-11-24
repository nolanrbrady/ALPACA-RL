"""Schema validation and column metadata management for ALPACA."""

from typing import Dict, List

from env_constants import DELTA_COLUMN, NO_MEDICATION_ACTION


class SchemaValidator:
    """Parses and validates the column schema configuration."""

    def __init__(self, schema: Dict[str, object]):
        self.schema = schema
        self.delta_col = DELTA_COLUMN
        
        # Extract column lists
        self.action_cols = list(self.schema.get('action_cols', []))
        self.observation_cols = list(self.schema.get('observation_cols', []))
        
        fallback_inputs = self.observation_cols + self.action_cols + [self.delta_col]
        self.model_input_cols = list(self.schema.get('model_input_cols', fallback_inputs))
        
        self.model_cont_output_cols = [c for c in self.schema.get('y_cont_cols', [])]
        self.model_binary_output_cols = [c for c in self.schema.get('y_bin_cols', [])]

        self.binary_obs_cols = [c for c in self.observation_cols if c in set(self.model_binary_output_cols)]
        self.cont_obs_cols = [c for c in self.observation_cols if c not in set(self.binary_obs_cols)]
        self.y_categorical_groups = self.schema.get('y_categorical_groups', {})
        
        # Validate immediately upon initialization
        self._validate_schema_columns()

    def _validate_schema_columns(self) -> None:
        """Validate schema-derived column lists to prevent downstream misalignment."""
        sequences = {
            'observation_cols': self.observation_cols,
            'action_cols': self.action_cols,
            'model_input_cols': self.model_input_cols,
            'model_cont_output_cols': self.model_cont_output_cols,
            'model_binary_output_cols': self.model_binary_output_cols,
        }
        for name, cols in sequences.items():
            duplicates = self._find_duplicate_columns(cols)
            if duplicates:
                raise ValueError(f"Schema list '{name}' contains duplicate entries: {duplicates}")

        if self.delta_col not in self.model_input_cols:
            raise ValueError(f"Model input columns must include time delta '{self.delta_col}'.")

        missing_obs = [c for c in self.observation_cols if c not in self.model_input_cols]
        if missing_obs:
            raise ValueError(
                "Observation columns missing from model_input_cols; expected all observations to be model inputs: "
                f"{missing_obs}"
            )
        missing_actions = [c for c in self.action_cols if c not in self.model_input_cols]
        if missing_actions:
            raise ValueError(
                "Action columns missing from model_input_cols; expected all actions to be model inputs: "
                f"{missing_actions}"
            )

    def _find_duplicate_columns(self, columns: List[str]) -> List[str]:
        """Return any duplicate column names while preserving insertion order."""
        seen = set()
        duplicates = []
        for col in columns:
            if col in seen and col not in duplicates:
                duplicates.append(col)
            seen.add(col)
        return duplicates

    def resolve_no_medication_index(self) -> int:
        """Locate the mutually-exclusive 'No Medication' action index."""
        try:
            return self.action_cols.index(NO_MEDICATION_ACTION)
        except ValueError as exc:
            raise ValueError(
                "Required action 'No Medication_active' missing from columns_schema.json. "
                "Regenerate preprocessing artifacts to restore the mutual-exclusion baseline."
            ) from exc

