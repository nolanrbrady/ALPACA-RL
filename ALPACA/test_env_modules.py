import json
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from ALPACA.artifact_loader import ArtifactLoader
from ALPACA.env_constants import DELTA_COLUMN, NO_MEDICATION_ACTION
from ALPACA.initial_state_sampler import InitialStateSampler
from ALPACA.reward_calculator import RewardCalculator
from ALPACA.scaler_validator import ScalerValidator
from ALPACA.state_validator import StateValidator


class DummyScaler:
    """Lightweight scaler stub for validator tests."""

    def __init__(self, names, mean=None, scale=None, var=None):
        self.feature_names_in_ = np.array(names, dtype=object)
        self.n_features_in_ = len(names)
        self.mean_ = np.array(mean if mean is not None else [0.0] * len(names), dtype=float)
        self.scale_ = np.array(scale if scale is not None else [1.0] * len(names), dtype=float)
        self.var_ = np.array(var if var is not None else [1.0] * len(names), dtype=float)


class IdentityTransformer:
    """Minimal transformer with inverse_transform."""

    def inverse_transform(self, arr):
        return np.asarray(arr, dtype=float)


def test_artifact_loader_missing_schema_raises(tmp_path):
    loader = ArtifactLoader(artifact_root=tmp_path)
    with pytest.raises(ValueError):
        loader.load_schema()


def test_artifact_loader_missing_no_med_action(monkeypatch):
    loader = ArtifactLoader()
    monkeypatch.setattr(loader, "load_scalers", lambda: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(loader, "load_variable_bounds", lambda: pd.DataFrame())
    monkeypatch.setattr(loader, "load_model", lambda *args, **kwargs: SimpleNamespace())

    bad_schema = {"action_cols": ["SomeOtherAction"], "model_input_cols": [DELTA_COLUMN]}
    monkeypatch.setattr(loader, "load_schema", lambda: bad_schema)
    with pytest.raises(ValueError, match="No Medication_active"):
        loader.load_all(input_dim=1, out_cont_dim=0, out_bin_dim=0, device="cpu")


def test_artifact_loader_missing_delta(monkeypatch):
    loader = ArtifactLoader()
    monkeypatch.setattr(loader, "load_scalers", lambda: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(loader, "load_variable_bounds", lambda: pd.DataFrame())
    monkeypatch.setattr(loader, "load_model", lambda *args, **kwargs: SimpleNamespace())

    bad_schema = {"action_cols": [NO_MEDICATION_ACTION], "model_input_cols": []}
    monkeypatch.setattr(loader, "load_schema", lambda: bad_schema)
    with pytest.raises(ValueError, match="time delta"):
        loader.load_all(input_dim=1, out_cont_dim=0, out_bin_dim=0, device="cpu")


def test_scaler_validator_alignment_and_duplicates():
    validator = ScalerValidator(
        model_input_cols=['a', DELTA_COLUMN, 'b'],
        cont_obs_cols=['a'],
        model_cont_output_cols=['y'],
    )
    scaler_X = DummyScaler(names=['months_since_bl', 'time_delta', 'drop_me', 'a'], mean=[0, 1, 2, 3])
    validator.align_scaler_X_feature_names(scaler_X)
    assert list(scaler_X.feature_names_in_) == [DELTA_COLUMN, 'a']
    assert list(scaler_X.mean_) == [1.0, 3.0]

    # Duplicate features should raise
    scaler_dup = DummyScaler(names=['a', 'a'])
    with pytest.raises(ValueError, match="duplicate"):
        validator.validate_alignment(scaler_dup, DummyScaler(names=['y']))


def test_scaler_validator_missing_inputs_and_outputs():
    validator = ScalerValidator(
        model_input_cols=['a', DELTA_COLUMN],
        cont_obs_cols=['a'],
        model_cont_output_cols=['y'],
    )
    scaler_X = DummyScaler(names=['a'])  # missing delta
    scaler_y = DummyScaler(names=['other'])  # missing y
    with pytest.raises(ValueError, match="missing required continuous inputs"):
        validator.validate_alignment(scaler_X, scaler_y)

    scaler_X = DummyScaler(names=['a', DELTA_COLUMN, 'extra'])
    with pytest.raises(ValueError, match="not present in model_input_cols"):
        validator.validate_alignment(scaler_X, scaler_y)

    scaler_X = DummyScaler(names=[DELTA_COLUMN, 'a'])
    with pytest.raises(ValueError, match="continuous outputs"):
        validator.validate_alignment(scaler_X, scaler_y)


def test_state_validator_bounds_and_categoricals():
    bounds = pd.DataFrame(
        {'lower_bound': [-1.0, 0.0], 'upper_bound': [1.0, 1.0]},
        index=['c1', 'cat_a'],
    )
    validator = StateValidator(
        observation_cols=['c1', 'cat_a', 'cat_b'],
        variable_bounds=bounds,
        y_categorical_groups={'cat': ['cat_a', 'cat_b']},
    )
    sample = pd.Series({'c1': 2.0, 'cat_a': 0.2, 'cat_b': 0.7})
    clipped = validator.clip_sample_to_bounds(sample.copy())
    assert clipped['c1'] == 1.0
    enforced = validator.enforce_categorical_groups(sample.copy())
    assert enforced['cat_b'] == 1.0 and enforced['cat_a'] == 0.0

    valid, violations = validator.check_state_bounds(np.array([0.0, 0.5, 0.5]))
    assert valid and violations == []
    valid, violations = validator.check_state_bounds(np.array([np.inf, 0.5, 0.5]))
    assert not valid and violations[0]['violation'] == 'non_finite'


def test_initial_state_sampler_errors_and_sampling():
    observation_cols = ['x1', 'cat_a']
    bounds = pd.DataFrame(
        {'lower_bound': [-5.0, 0.0], 'upper_bound': [5.0, 1.0]},
        index=['x1', 'cat_a'],
    )
    y_groups = {'cat': ['cat_a']}
    valid_gaussians = {
        'observation_cols': observation_cols,
        'distributions': {
            'all': {
                'clusters': [{'mean': np.array([0.0, 0.0]), 'cov': np.eye(2), 'weight': 1.0}],
                'weights': np.array([1.0]),
                'power_transformer': IdentityTransformer(),
                'num_samples': 1,
            }
        }
    }
    sampler = InitialStateSampler(
        observation_cols=observation_cols,
        y_categorical_groups=y_groups,
        variable_bounds=bounds,
        initial_state_payload=valid_gaussians,
    )
    rng = np.random.default_rng(0)
    sample = sampler.sample('all', np_random=rng)
    assert sample.shape == (2,)
    assert 0.0 <= sample[1] <= 1.0  # categorical clipped/one-hot

    bad_gaussians = {}
    with pytest.raises(ValueError):
        sampler_bad = InitialStateSampler(
            observation_cols=observation_cols,
            y_categorical_groups=y_groups,
            variable_bounds=bounds,
            initial_state_payload=bad_gaussians,
        )

    missing_transform = {
        'observation_cols': observation_cols,
        'distributions': {
            'all': {
                'clusters': [{'mean': np.array([0.0, 0.0]), 'cov': np.eye(2), 'weight': 1.0}],
                'weights': np.array([1.0]),
                'power_transformer': None,
                'num_samples': 1,
            }
        }
    }
    with pytest.raises(ValueError):
        sampler_missing = InitialStateSampler(
            observation_cols=observation_cols,
            y_categorical_groups=y_groups,
            variable_bounds=bounds,
            initial_state_payload=missing_transform,
        )


def make_reward_calc(reliability=0.9):
    return RewardCalculator(reward_metric='metric', reliability_rxx=reliability)


def test_reward_calculator_basic_and_bounds():
    calc = make_reward_calc()
    prev = (0.0, 1.0)
    nxt = (1.0, 1.0)
    obs_cols = ['metric', 'other']

    reward = calc.calculate_reward(prev, nxt, obs_cols, bounds_check=lambda s: (True, []))
    # reliable change with r_xx=0.9, SD=1 -> s_diff ~ sqrt(0.2)
    expected = float(np.clip(10.0 * (1.0 / np.sqrt(0.2)), -10.0, 10.0))
    assert reward == pytest.approx(expected)

    reward = calc.calculate_reward(prev, nxt, obs_cols, bounds_check=lambda s: (False, []))
    assert reward == 0.0


def test_reward_calculator_missing_metric_and_small_sdiff():
    calc = make_reward_calc(reliability=1.0)
    prev = (0.0, 1.0)
    nxt = (1.0, 2.0)
    obs_cols = ['other', 'other2']
    assert calc.calculate_reward(prev, nxt, obs_cols, bounds_check=lambda s: (True, [])) == 0.0

    calc = make_reward_calc(reliability=0.999999)
    obs_cols = ['metric', 'other']
    reward = calc.calculate_reward(prev, nxt, obs_cols, bounds_check=lambda s: (True, []))
    assert np.isfinite(reward)
