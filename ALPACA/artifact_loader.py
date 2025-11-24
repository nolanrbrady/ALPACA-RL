"""Utilities for locating and loading ALPACA environment artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import pandas as pd
import torch

from .env_constants import DELTA_COLUMN, NO_MEDICATION_ACTION
from .moe_transformer import TransformerWithMoE


@dataclass
class ArtifactPaths:
    """Resolved file paths for ALPACA artifacts."""

    scaler_x_path: Path
    scaler_y_path: Path
    model_path: Path
    bounds_path: Path
    schema_path: Path
    initial_state_gaussians_path: Optional[Path]


@dataclass
class LoadedArtifacts:
    """Container of loaded ALPACA artifacts."""

    scaler_X: object
    scaler_y: object
    variable_bounds: pd.DataFrame
    schema: Dict[str, object]
    initial_state_gaussians: Optional[Dict[str, object]]
    model: TransformerWithMoE


class ArtifactLoader:
    """Resolve and load all ALPACA artifacts with consistent error handling."""

    def __init__(
        self,
        artifact_root: Optional[Union[str, Path]] = None,
        initial_state_gaussian_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Create an artifact loader.

        Args:
            artifact_root: Directory containing ALPACA artifacts. Defaults to the package folder.
            initial_state_gaussian_path: Optional path to a custom Gaussian initializer artifact.
        """
        self.artifact_root = Path(artifact_root) if artifact_root is not None else Path(__file__).resolve().parent
        self.artifact_root = self.artifact_root.resolve()
        if not self.artifact_root.is_dir():
            raise ValueError(f"Artifact root '{self.artifact_root}' does not exist or is not a directory.")

        resolved_gaussians = (
            Path(initial_state_gaussian_path).resolve()
            if initial_state_gaussian_path is not None
            else self.artifact_root / 'initial_state_gaussians.joblib'
        )
        self.paths = self._resolve_paths(resolved_gaussians)

    def _resolve_paths(self, gaussian_path: Optional[Path]) -> ArtifactPaths:
        """Resolve paths relative to artifact_root."""
        return ArtifactPaths(
            scaler_x_path=self.artifact_root / 'scaler_X.joblib',
            scaler_y_path=self.artifact_root / 'scaler_y.joblib',
            model_path=self.artifact_root / 'best_moe_transformer_model.pt',
            bounds_path=self.artifact_root / 'ADNI_Variable_Bounds.csv',
            schema_path=self.artifact_root / 'columns_schema.json',
            initial_state_gaussians_path=gaussian_path,
        )

    def load_scalers(self) -> tuple[object, object]:
        """Load feature and target scalers used during model training."""
        scaler_X = joblib.load(self.paths.scaler_x_path)
        scaler_y = joblib.load(self.paths.scaler_y_path)
        return scaler_X, scaler_y

    def load_variable_bounds(self) -> pd.DataFrame:
        """Load ADNI variable bounds used for observation space validation."""
        try:
            return pd.read_csv(self.paths.bounds_path, index_col=0)
        except Exception as exc:
            raise ValueError(f"Failed to read variable bounds from {self.paths.bounds_path}: {exc}") from exc

    def load_schema(self) -> Dict[str, object]:
        """Read the preprocessing schema to recover ordered column definitions."""
        try:
            with open(self.paths.schema_path, 'r') as f:
                return json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to read columns_schema.json from {self.paths.schema_path}: {exc}") from exc

    def load_model(
        self,
        input_dim: int,
        out_cont_dim: int,
        out_bin_dim: int,
        device: torch.device,
    ) -> TransformerWithMoE:
        """Instantiate the Transformer model and load trained weights."""
        model = TransformerWithMoE(
            input_dim=input_dim,
            out_cont_dim=out_cont_dim,
            out_bin_dim=out_bin_dim,
        ).to(device)
        try:
            state = torch.load(self.paths.model_path, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from {self.paths.model_path}: {exc}") from exc
        model.eval()
        return model

    def load_initial_state_gaussians(self) -> Optional[Dict[str, Dict[str, object]]]:
        """Load precomputed Gaussian parameters for initial states if available."""
        path = self.paths.initial_state_gaussians_path
        if path is None or not path.is_file():
            return None
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"Failed to read initial state artifact from {path}: {exc}") from exc
        return payload

    def load_all(
        self,
        input_dim: int,
        out_cont_dim: int,
        out_bin_dim: int,
        device: torch.device,
    ) -> LoadedArtifacts:
        """Load all artifacts and return them as a bundle."""
        scaler_X, scaler_y = self.load_scalers()
        variable_bounds = self.load_variable_bounds()
        schema = self.load_schema()
        initial_state_gaussians = self.load_initial_state_gaussians()

        if schema.get('action_cols') and NO_MEDICATION_ACTION not in schema.get('action_cols', []):
            raise ValueError(
                "Required action 'No Medication_active' missing from columns_schema.json. "
                "Regenerate preprocessing artifacts to restore the mutual-exclusion baseline."
            )
        if DELTA_COLUMN not in schema.get('model_input_cols', []):
            raise ValueError(f"Model input columns must include time delta '{DELTA_COLUMN}'.")

        model = self.load_model(
            input_dim=input_dim,
            out_cont_dim=out_cont_dim,
            out_bin_dim=out_bin_dim,
            device=device,
        )
        return LoadedArtifacts(
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            variable_bounds=variable_bounds,
            schema=schema,
            initial_state_gaussians=initial_state_gaussians,
            model=model,
        )
