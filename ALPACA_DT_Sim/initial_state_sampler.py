"""Gaussian initial state sampling for ALPACA."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .state_validator import StateValidator


class InitialStateSampler:
    """Sample initial patient states from precomputed Gaussian mixtures."""

    def __init__(
        self,
        observation_cols,
        y_categorical_groups,
        variable_bounds,
        initial_state_payload: Optional[Dict[str, object]],
    ):
        self.observation_cols = list(observation_cols)
        self.y_categorical_groups = y_categorical_groups
        self.variable_bounds = variable_bounds
        
        self.state_validator = StateValidator(
            observation_cols=self.observation_cols,
            variable_bounds=self.variable_bounds,
            y_categorical_groups=self.y_categorical_groups,
        )
        
        # Validate and parse the raw payload immediately upon initialization
        self.initial_state_gaussians = self._validate_initial_state_gaussians(initial_state_payload)

    def sample(self, cohort_type: str, np_random=None) -> np.ndarray:
        """Sample an initial state from the Gaussian distribution for the cohort."""
        if not self.initial_state_gaussians:
            raise ValueError(
                "Initial state Gaussian artifact not found or invalid. "
                "Run build_initial_state_gaussians.py to generate the artifact."
            )
        cohort_stats = self.initial_state_gaussians.get(cohort_type)
        if cohort_stats is None:
            raise ValueError(f"Cohort '{cohort_type}' not found in initial state gaussians.")
        clusters = cohort_stats.get('clusters', [])
        if not clusters:
            raise ValueError(f"Cohort '{cohort_type}' is missing Gaussian mixture clusters.")
        weights = cohort_stats.get('weights')
        if weights is None or len(weights) != len(clusters):
            raise ValueError(f"Cohort '{cohort_type}' is missing valid mixture weights.")
        rng = np_random
        if rng is None or not hasattr(rng, 'multivariate_normal'):
            rng = np.random.default_rng()

        cluster_idx = int(rng.choice(len(clusters), p=weights))
        cluster = clusters[cluster_idx]
        sample = rng.multivariate_normal(cluster['mean'], cluster['cov'])
        power_transformer = cohort_stats.get('power_transformer')
        if power_transformer is None or not hasattr(power_transformer, 'inverse_transform'):
            raise ValueError(
                f"Cohort '{cohort_type}' is missing a valid PowerTransformer. "
                "Regenerate the Gaussian artifact to refresh serialized transformers."
            )
        sample = power_transformer.inverse_transform(sample.reshape(1, -1))[0]

        sample_series = pd.Series(sample, index=self.observation_cols, dtype=np.float32)
        sample_series = self.state_validator.enforce_categorical_groups(sample_series)
        sample_series = self.state_validator.clip_sample_to_bounds(sample_series)
        return sample_series.values.astype(np.float32)

    def _validate_initial_state_gaussians(self, payload: Optional[Dict[str, object]]) -> Optional[Dict[str, Dict[str, object]]]:
        """Load and validate Gaussian parameters for initial states."""
        if payload is None:
            return None
        
        # Verify schema alignment
        artifact_cols = payload.get('observation_cols', [])
        if artifact_cols != self.observation_cols:
            raise ValueError(
                "Observation columns in initial_state_gaussians.joblib do not match columns_schema.json. "
                "Regenerate the artifact after updating preprocessing artifacts."
            )
            
        distributions = payload.get('distributions', {})
        loaded: Dict[str, Dict[str, object]] = {}
        
        for cohort_name, stats in distributions.items():
            clusters = []
            weights = []
            for idx, cluster in enumerate(stats.get('clusters', [])):
                mean = np.asarray(cluster.get('mean', []), dtype=float)
                cov = np.asarray(cluster.get('cov', []), dtype=float)

                if mean.shape[0] != len(self.observation_cols):
                    raise ValueError(f"Cluster {idx} mean for cohort '{cohort_name}' has invalid length.")
                if cov.shape != (len(self.observation_cols), len(self.observation_cols)):
                    raise ValueError(f"Cluster {idx} covariance for cohort '{cohort_name}' has invalid shape.")

                clusters.append({'mean': mean, 'cov': cov})
                weights.append(float(cluster.get('weight', 0.0)))
            
            if not clusters:
                raise ValueError(f"Cohort '{cohort_name}' is missing Gaussian mixture clusters.")
                
            weights_arr = np.asarray(weights, dtype=float)
            if np.any(weights_arr < 0):
                raise ValueError(f"Cohort '{cohort_name}' has negative mixture weights.")
            
            weight_sum = float(weights_arr.sum())
            if weight_sum <= 0.0:
                raise ValueError(f"Cohort '{cohort_name}' mixture weights sum to zero.")
            
            weights_arr = weights_arr / weight_sum
            power_transformer = stats.get('power_transformer')
            
            if power_transformer is None or not hasattr(power_transformer, 'inverse_transform'):
                raise ValueError(
                    f"Cohort '{cohort_name}' is missing a valid PowerTransformer. "
                    "Regenerate the Gaussian artifact to refresh serialized transformers."
                )
                
            loaded[cohort_name] = {
                'clusters': clusters,
                'weights': weights_arr,
                'power_transformer': power_transformer,
                'num_samples': int(stats.get('num_samples', 0)),
            }
            
        return loaded if loaded else None
