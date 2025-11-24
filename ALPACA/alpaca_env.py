"""ALzheimers Prophelactic Action Control Agent (ALPACA) Gymnasium environment."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from artifact_loader import ArtifactLoader
from env_constants import (
    DEFAULT_MAX_EPISODE_LENGTH,
    DEFAULT_RELIABILITY_RXX,
    DELTA_COLUMN,
    NO_MEDICATION_ACTION,
    SUPPORTED_COHORTS,
)
from initial_state_sampler import InitialStateSampler
from reward_calculator import RewardCalculator
from scaler_validator import ScalerValidator
from schema_validator import SchemaValidator
from state_scaling import manage_state_scaling as shared_manage_state_scaling
from state_validator import StateValidator
from action_validator import ActionValidator


class ALPACAEnv(gym.Env):
    """Gymnasium environment backed by the pretrained ALPACA dynamics model."""

    def __init__(
        self,
        time_delta_months: float = 6.0,
        reward_metric: str = 'ADNI_MEM',
        mc_samples: int = 0,
        cohort_type: str = 'impaired',
        artifact_loader: Optional[ArtifactLoader] = None,
    ):
        """
        Initialize the ALPACA environment.

        Args:
            time_delta_months (float): Fixed step size in months passed to model as next_visit_months.
            reward_metric (str): Column name in observation used for reward delta (default: 'ADNI_MEM').
            mc_samples (int): Number of MC-dropout samples for uncertainty estimation.
            cohort_type (str): Cohort subset to sample starts from ('all', 'impaired', 'healthy').
            artifact_loader (ArtifactLoader | None): Optional artifact loader for injecting custom paths.
        """
        super().__init__()
        self.reward_metric = reward_metric
        self.mc_samples = int(mc_samples) if mc_samples is not None else 0
        self.cohort_type = self._validate_cohort_type(cohort_type)
        self.reliability_rxx = DEFAULT_RELIABILITY_RXX

        loader = artifact_loader if artifact_loader is not None else ArtifactLoader()
        self.artifact_loader = loader
        self.schema_validator = SchemaValidator(loader.load_schema())
        self.schema = self.schema_validator.schema

        self.delta_col = self.schema_validator.delta_col
        self.action_cols = self.schema_validator.action_cols
        self.observation_cols = self.schema_validator.observation_cols
        self.model_input_cols = self.schema_validator.model_input_cols
        self.model_cont_output_cols = self.schema_validator.model_cont_output_cols
        self.model_binary_output_cols = self.schema_validator.model_binary_output_cols
        self.binary_obs_cols = self.schema_validator.binary_obs_cols
        self.cont_obs_cols = self.schema_validator.cont_obs_cols
        self.y_categorical_groups = self.schema_validator.y_categorical_groups

        self.scaler_X, self.scaler_y = loader.load_scalers()
        self.scaler_validator = ScalerValidator(
            model_input_cols=self.model_input_cols,
            cont_obs_cols=self.cont_obs_cols,
            model_cont_output_cols=self.model_cont_output_cols,
        )
        self.scaler_validator.align_scaler_X_feature_names(self.scaler_X)

        self.variable_bounds = loader.load_variable_bounds()
        self.state_validator = StateValidator(
            observation_cols=self.observation_cols,
            variable_bounds=self.variable_bounds,
            y_categorical_groups=self.y_categorical_groups,
        )
        self._validate_continuous_bounds()
        self.scaler_validator.validate_alignment(self.scaler_X, self.scaler_y)

        raw_gaussians = loader.load_initial_state_gaussians()
        self.initial_state_sampler = InitialStateSampler(
            observation_cols=self.observation_cols,
            y_categorical_groups=self.y_categorical_groups,
            variable_bounds=self.variable_bounds,
            initial_state_payload=raw_gaussians,
        )
        self.initial_state_gaussians = self.initial_state_sampler.initial_state_gaussians
        
        if not self.initial_state_sampler.initial_state_gaussians:
            raise ValueError(
                "Initial state Gaussian artifact not found or invalid. "
                f"Expected parameters at '{loader.paths.initial_state_gaussians_path}'. "
                "Run build_initial_state_gaussians.py to generate the artifact."
            )

        self._no_med_idx = self.schema_validator.resolve_no_medication_index()
        self.action_validator = ActionValidator(
            action_cols=self.action_cols,
            no_med_idx=self._no_med_idx
        )

        self.device = self._select_device()
        self.model = loader.load_model(
            input_dim=len(self.model_input_cols),
            out_cont_dim=len(self.model_cont_output_cols),
            out_bin_dim=len(self.model_binary_output_cols),
            device=self.device,
        )
        self.reward_calculator = RewardCalculator(self.reward_metric, self.reliability_rxx)

        self.max_episode_length = DEFAULT_MAX_EPISODE_LENGTH
        self.time_delta_val = float(time_delta_months)
        self.current_step = 0
        self.terminated = False
        self.truncated = False
        self.reward = 0.0
        self.info: Dict[str, object] = {}
        self._seq_inputs: list[np.ndarray] = []

        self.action_space = gym.spaces.MultiBinary(len(self.action_cols))
        self.observation_space = self._build_observation_space()

        self.state = self.reset()[0]

    @classmethod
    def from_artifacts(
        cls,
        artifact_root: Union[str, Path],
        *,
        initial_state_gaussian_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "ALPACAEnv":
        """Construct an environment using artifacts rooted at the provided path."""
        loader = ArtifactLoader(
            artifact_root=artifact_root,
            initial_state_gaussian_path=initial_state_gaussian_path,
        )
        return cls(artifact_loader=loader, **kwargs)

    def _validate_cohort_type(self, cohort_type: str) -> str:
        """Ensure the provided cohort is supported."""
        if cohort_type not in SUPPORTED_COHORTS:
            raise ValueError(
                f"Invalid cohort type: {cohort_type}. Must be one of: {', '.join(SUPPORTED_COHORTS)}."
            )
        return cohort_type

    def _select_device(self) -> torch.device:
        """Choose the most capable torch device available."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_observation_space(self) -> gym.spaces.Box:
        """Construct the Gymnasium observation space backed by ADNI bounds."""
        lows = np.zeros(len(self.observation_cols), dtype=np.float32)
        highs = np.zeros(len(self.observation_cols), dtype=np.float32)
        for i, col in enumerate(self.observation_cols):
            if self.variable_bounds is not None and col in self.variable_bounds.index:
                lows[i] = float(self.variable_bounds.loc[col, 'lower_bound'])
                highs[i] = float(self.variable_bounds.loc[col, 'upper_bound'])
            else:
                lows[i] = 0.0
                highs[i] = 1.0
        return gym.spaces.Box(low=lows, high=highs, shape=(len(self.observation_cols),), dtype=np.float32)

    def _build_model_input(self, current_obs_series: pd.Series, action: np.ndarray) -> pd.DataFrame:
        """Assemble and scale the autoregressive model input row for the next prediction."""
        action_series = pd.Series(action, index=self.action_cols)
        model_input_series = pd.concat([current_obs_series, action_series])
        model_input_series[self.delta_col] = self.time_delta_val
        model_input_df = pd.DataFrame([model_input_series])[self.model_input_cols]
        self._validate_model_input_frame(model_input_df)

        cols_to_scale = [c for c in getattr(self.scaler_X, 'feature_names_in_', []) if c in model_input_df.columns]
        if cols_to_scale:
            scaled_subset = self.manage_state_scaling(model_input_df[cols_to_scale], self.scaler_X, normalize=True)
            for c in cols_to_scale:
                model_input_df[c] = np.float32(scaled_subset[c].values)
        self._validate_model_input_frame(model_input_df)
        return model_input_df

    def _validate_model_input_frame(self, model_input_df: pd.DataFrame) -> None:
        """Ensure model inputs contain all required columns with finite numeric values."""
        missing = [c for c in self.model_input_cols if c not in model_input_df.columns]
        if missing:
            raise ValueError(f"Model input is missing required columns: {missing}")
        row = model_input_df.iloc[0]
        non_finite = [c for c, v in row.items() if not np.isfinite(v)]
        if non_finite:
            raise ValueError(f"Model input contains non-finite values for columns: {non_finite}")

    def _predict_next_step(
        self, sequence_tensor: torch.Tensor, attn_mask: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray], Optional[np.ndarray]]:
        """Run the MoE Transformer (with optional MC dropout) and return final-step predictions."""
        if isinstance(self.mc_samples, int) and self.mc_samples > 0:
            prev_training = self.model.training
            cont_samples: list[np.ndarray] = []
            bin_samples: list[np.ndarray] = []
            try:
                self.model.train(True)
                with torch.no_grad():
                    for _ in range(self.mc_samples):
                        pred_cont, pred_bin = self.model(sequence_tensor, attn_mask=attn_mask)
                        if pred_cont is not None:
                            cont_samples.append(pred_cont[:, -1, :].detach().cpu().numpy()[0])
                        if pred_bin is not None:
                            bin_logits = pred_bin[:, -1, :].detach().cpu().numpy()[0]
                            bin_samples.append(bin_logits)
                cont_arr = np.stack(cont_samples, axis=0) if cont_samples else None
                bin_arr = np.stack(bin_samples, axis=0) if bin_samples else None
                last_cont = None
                last_bin = None
                cont_samples_unscaled = None
                bin_prob_samples = None
                if cont_arr is not None and cont_arr.size > 0:
                    cont_mean_scaled = np.mean(cont_arr, axis=0)
                    last_cont = torch.tensor(cont_mean_scaled.reshape(1, -1), dtype=torch.float32, device=self.device)
                    cont_samples_unscaled = []
                    for sample in cont_arr:
                        s = pd.Series(sample, index=self.model_cont_output_cols)
                        cont_samples_unscaled.append(
                            self._inverse_scale_series(s, self.scaler_y, self.model_cont_output_cols).values
                        )
                    cont_samples_unscaled = np.stack(cont_samples_unscaled, axis=0)
                if bin_arr is not None and bin_arr.size > 0:
                    bin_mean_logits = np.mean(bin_arr, axis=0)
                    last_bin = torch.tensor(bin_mean_logits.reshape(1, -1), dtype=torch.float32, device=self.device)
                    bin_prob_samples = 1.0 / (1.0 + np.exp(-bin_arr))
            finally:
                self.model.train(prev_training)
            return last_cont, last_bin, cont_samples_unscaled, bin_prob_samples

        pred_cont, pred_bin = self.model(sequence_tensor, attn_mask=attn_mask)
        last_cont = pred_cont[:, -1, :] if pred_cont is not None else None
        last_bin = pred_bin[:, -1, :] if pred_bin is not None else None
        return last_cont, last_bin, None, None

    def _apply_predictions_to_observation(
        self,
        current_obs_series: pd.Series,
        cont_scaled_series: pd.Series,
        bin_series: pd.Series,
    ) -> pd.Series:
        """Merge model outputs into a new observation while enforcing categorical constraints."""
        next_obs_series = current_obs_series.copy()
        cont_unscaled = self._inverse_scale_series(cont_scaled_series, self.scaler_y, self.model_cont_output_cols)
        for col, value in cont_unscaled.items():
            if col in next_obs_series.index:
                next_obs_series[col] = np.float32(value)

        used_bin_cols = set()
        for group_cols in getattr(self, 'y_categorical_groups', {}).values():
            cols = [c for c in group_cols if c in self.model_binary_output_cols]
            if not cols:
                continue
            probs = bin_series[cols].astype(float)
            if len(probs) == 0:
                continue
            winner = probs.idxmax()
            for c in cols:
                if c in next_obs_series.index:
                    next_obs_series[c] = 1.0 if c == winner else 0.0
            used_bin_cols.update(cols)
        for c, p in bin_series.items():
            if c not in used_bin_cols and c in next_obs_series.index:
                next_obs_series[c] = float(np.round(p))

        if 'subject_age' in next_obs_series.index:
            updated_age = float(current_obs_series.get('subject_age', 0.0)) + (self.time_delta_val / 12.0)
            next_obs_series['subject_age'] = np.float32(updated_age)
        return next_obs_series

    def _record_uncertainty_metrics(
        self,
        cont_samples_unscaled: Optional[np.ndarray],
        bin_samples: Optional[np.ndarray],
    ) -> None:
        """Populate self.info with MC-dropout uncertainty summaries (if available)."""
        if not isinstance(self.mc_samples, int) or self.mc_samples <= 0:
            return
        if cont_samples_unscaled is not None and cont_samples_unscaled.size > 0:
            cont_std = np.std(cont_samples_unscaled, axis=0)
            self.info['mean_cont_uncertainty'] = float(np.mean(cont_std))
            metric = self.reward_metric
            if metric in self.model_cont_output_cols:
                idx = self.model_cont_output_cols.index(metric)
                metric_samples = cont_samples_unscaled[:, idx]
                lower = float(np.percentile(metric_samples, 2.5))
                upper = float(np.percentile(metric_samples, 97.5))
                self.info[f"{metric.lower()}_ci_lower"] = lower
                self.info[f"{metric.lower()}_ci_upper"] = upper
                self.info[f"{metric.lower()}_uncertainty"] = float(np.std(metric_samples))
        if bin_samples is not None and bin_samples.size > 0 and hasattr(self, 'model_binary_output_cols'):
            bin_std = np.std(bin_samples, axis=0)
            self.info['mean_bin_uncertainty'] = float(np.mean(bin_std))

    def _validate_continuous_bounds(self) -> None:
        """Ensure ADNI bounds cover every continuous observation column."""
        if self.variable_bounds is None:
            raise ValueError("Variable bounds DataFrame is not initialized.")
        missing = [col for col in self.cont_obs_cols if col not in self.variable_bounds.index]
        if missing:
            raise ValueError(
                "ADNI_Variable_Bounds.csv is missing required continuous features: "
                f"{missing}. Regenerate preprocessing artifacts to refresh the bounds."
            )

    def _inverse_scale_series(self, series: pd.Series, scaler, cols: list[str]) -> pd.Series:
        """Inverse-scale values for specified cols using scaler stats (per-column)."""
        out = series.copy()
        if not hasattr(scaler, 'feature_names_in_'):
            return out
        name_to_idx = {c: i for i, c in enumerate(scaler.feature_names_in_)}
        for c in cols:
            if c in name_to_idx and c in out.index:
                j = name_to_idx[c]
                out[c] = np.float32(out[c] * scaler.scale_[j] + scaler.mean_[j])
        return out

    def manage_state_scaling(self, state_data, scaler, normalize=True):
        """Delegate to the shared scaling helper for backward compatibility."""
        return shared_manage_state_scaling(state_data, scaler, normalize=normalize)

    def get_start_state(self):
        """Sample the initial patient state from the configured Gaussian mixture distribution."""
        gaussian_state = self.initial_state_sampler.sample(self.cohort_type, np_random=getattr(self, 'np_random', None))
        return gaussian_state

    def check_state_bounds(self, state_values):
        """Check if state values are within the acceptable ADNI variable bounds."""
        return self.state_validator.check_state_bounds(state_values)

    def reset(self, seed=None, options=None):
        """Reset the environment and resample the Gaussian initial patient state."""
        super().reset(seed=seed)
        self.current_step = 0
        self._seq_inputs = []
        self.state = self.get_start_state()
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.info = {}
        return self.state, self.info

    def calculate_reward(self, prev_obs, next_obs):
        """Calculate reliable-change reward for the configured metric."""
        return self.reward_calculator.calculate_reward(
            prev_obs=prev_obs,
            next_obs=next_obs,
            observation_cols=self.observation_cols,
            bounds_check=self.state_validator.check_state_bounds,
        )

    def step(self, action):
        """Advance the ALPACA patient simulator by one treatment decision."""
        if self.terminated or self.truncated:
            return self.state, self.reward, bool(self.terminated), bool(self.truncated), self.info

        if not np.isfinite(self.state).all():
            raise ValueError("Current state contains non-finite values.")

        self.info = {}
        self.action_validator.validate_action_structure(action)
        constraint = self.action_validator.check_constraints(action, len(self._seq_inputs))
        if constraint is not None:
            self.terminated = True
            self.reward, self.info = constraint
            return self.state, self.reward, True, False, self.info

        self.current_step += 1
        current_obs_series = pd.Series(self.state, index=self.observation_cols)
        model_input_df = self._build_model_input(current_obs_series, action)
        self._validate_model_input_frame(model_input_df)

        self._seq_inputs.append(model_input_df.values.astype(np.float32)[0])
        sequence = torch.tensor(
            np.stack(self._seq_inputs, axis=0)[None, ...],
            dtype=torch.float32,
            device=self.device,
        )
        seq_len = sequence.shape[1]
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device), diagonal=1)

        last_cont, last_bin, cont_samples_unscaled, bin_samples = self._predict_next_step(sequence, attn_mask)

        if last_cont is not None:
            cont_vals = last_cont.detach().cpu().numpy()[0]
            cont_scaled_series = pd.Series(cont_vals, index=self.model_cont_output_cols)
        else:
            cont_scaled_series = pd.Series(dtype=float)
        if last_bin is not None:
            bin_probs = torch.sigmoid(last_bin).detach().cpu().numpy()[0]
            bin_series = pd.Series(bin_probs, index=self.model_binary_output_cols)
        else:
            bin_series = pd.Series(dtype=float)

        if not cont_scaled_series.empty and not np.isfinite(cont_scaled_series.values).all():
            raise ValueError("Model continuous outputs contain non-finite values.")
        if not bin_series.empty and not np.isfinite(bin_series.values).all():
            raise ValueError("Model binary outputs contain non-finite values.")

        next_obs_series = self._apply_predictions_to_observation(current_obs_series, cont_scaled_series, bin_series)
        next_state_np = next_obs_series[self.observation_cols].values
        is_within_bounds, out_of_bounds_vars = self.state_validator.check_state_bounds(next_state_np)

        if not is_within_bounds:
            self.terminated = True
            self.reward = 0.0
            self.info = {
                'termination_reason': 'state_out_of_bounds',
                'out_of_bounds_variables': out_of_bounds_vars,
            }
            return self.state, self.reward, True, False, self.info

        self.reward = self.calculate_reward(self.state, next_state_np)
        self.state = next_state_np
        self.info = {'sequence_length': len(self._seq_inputs)}
        try:
            self._record_uncertainty_metrics(cont_samples_unscaled, bin_samples)
        except Exception:
            pass

        self.truncated = self.current_step >= self.max_episode_length

        return self.state, self.reward, False, bool(self.truncated), self.info

    def render(self, render_mode='None'):
        """Render hook (not implemented)."""
        pass

    def close(self):
        """Close hook (not implemented)."""
        pass
