import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from ALPACA_DT_Sim.action_validator import ActionValidator
from ALPACA_DT_Sim.artifact_loader import ArtifactLoader
from ALPACA_DT_Sim.env_constants import DELTA_COLUMN, NO_MEDICATION_ACTION
from ALPACA_DT_Sim.initial_state_sampler import InitialStateSampler
from ALPACA_DT_Sim.reward_calculator import RewardCalculator
from ALPACA_DT_Sim.scaler_validator import ScalerValidator
from ALPACA_DT_Sim.schema_validator import SchemaValidator
from ALPACA_DT_Sim.state_scaling import manage_state_scaling
from ALPACA_DT_Sim.state_validator import StateValidator


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


def test_artifact_loader_invalid_root_raises(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="does not exist"):
        ArtifactLoader(artifact_root=missing_dir)


def test_artifact_loader_invalid_bounds_csv_raises(tmp_path: Path):
    """Invalid bounds CSV should raise a ValueError with context."""
    (tmp_path / 'ADNI_Variable_Bounds.csv').write_text("", encoding="utf-8")
    loader = ArtifactLoader(artifact_root=tmp_path)
    with pytest.raises(ValueError, match="Failed to read variable bounds"):
        loader.load_variable_bounds()


def test_artifact_loader_invalid_gaussian_joblib_raises(tmp_path: Path):
    """Invalid Gaussian artifact should raise a ValueError with context."""
    (tmp_path / 'initial_state_gaussians.joblib').write_bytes(b"not-a-joblib")
    loader = ArtifactLoader(artifact_root=tmp_path)
    with pytest.raises(ValueError, match="Failed to read initial state artifact"):
        loader.load_initial_state_gaussians()


def test_artifact_loader_load_all_happy_path(monkeypatch: pytest.MonkeyPatch):
    """load_all should succeed when schema invariants are satisfied."""
    loader = ArtifactLoader()
    monkeypatch.setattr(loader, "load_scalers", lambda: (SimpleNamespace(), SimpleNamespace()))
    monkeypatch.setattr(loader, "load_variable_bounds", lambda: pd.DataFrame())
    monkeypatch.setattr(loader, "load_initial_state_gaussians", lambda: None)
    monkeypatch.setattr(loader, "load_model", lambda *args, **kwargs: SimpleNamespace())

    good_schema = {
        "action_cols": [NO_MEDICATION_ACTION],
        "model_input_cols": [DELTA_COLUMN],
    }
    monkeypatch.setattr(loader, "load_schema", lambda: good_schema)

    artifacts = loader.load_all(input_dim=1, out_cont_dim=0, out_bin_dim=0, device="cpu")
    assert artifacts.schema == good_schema


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

    # align_scaler_X_feature_names should early-return if no feature_names_in_
    validator.align_scaler_X_feature_names(SimpleNamespace())

    # ensure duplicate feature_names_in_ are de-duped by aligner (keeps first)
    scaler_X2 = DummyScaler(names=['a', 'a', 'time_since_prev'], mean=[1, 2, 3])
    validator.align_scaler_X_feature_names(scaler_X2)
    assert list(scaler_X2.feature_names_in_) == ['a', DELTA_COLUMN]
    assert list(scaler_X2.mean_) == [1.0, 3.0]


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

    # None scalers should fail fast
    with pytest.raises(ValueError, match="Both scaler_X and scaler_y"):
        validator.validate_alignment(None, scaler_y)

    # Empty feature_names_in_ should raise
    empty = DummyScaler(names=[])
    with pytest.raises(ValueError, match="missing feature_names_in_"):
        validator.validate_alignment(empty, DummyScaler(names=['y']))

    # DELTA_COLUMN enforced even if not part of model_input_cols
    validator_no_delta = ScalerValidator(
        model_input_cols=['a'],
        cont_obs_cols=['a'],
        model_cont_output_cols=['y'],
    )
    with pytest.raises(ValueError, match="must include the time-delta"):
        validator_no_delta.validate_alignment(DummyScaler(names=['a']), DummyScaler(names=['y']))

    # scaler_y missing feature_names_in_ should raise
    sy_missing_names = SimpleNamespace()
    with pytest.raises(ValueError, match="scaler_y is missing feature_names_in_"):
        validator.validate_alignment(DummyScaler(names=['a', DELTA_COLUMN]), sy_missing_names)

    # mismatched stats length should raise from _validate_scaler_stats_length
    bad_stats = DummyScaler(names=['a', DELTA_COLUMN], mean=[0.0])  # wrong length
    with pytest.raises(ValueError, match="does not match number of features"):
        validator.validate_alignment(bad_stats, DummyScaler(names=['y']))


def test_schema_validator_resolve_no_medication_index_raises():
    schema = {"action_cols": ["SomethingElse"], "observation_cols": [], "y_cont_cols": [], "y_bin_cols": []}
    v = SchemaValidator(schema)
    with pytest.raises(ValueError, match="No Medication_active"):
        v.resolve_no_medication_index()


def test_schema_validator_duplicate_columns_raise():
    schema = {
        "observation_cols": ["x", "x"],
        "action_cols": [NO_MEDICATION_ACTION],
        "model_input_cols": ["x", NO_MEDICATION_ACTION, DELTA_COLUMN],
        "y_cont_cols": [],
        "y_bin_cols": [],
    }
    with pytest.raises(ValueError, match="duplicate"):
        SchemaValidator(schema)


def test_schema_validator_missing_obs_or_actions_in_model_inputs_raise():
    schema_missing_obs = {
        "observation_cols": ["x"],
        "action_cols": [NO_MEDICATION_ACTION],
        "model_input_cols": [NO_MEDICATION_ACTION, DELTA_COLUMN],
        "y_cont_cols": [],
        "y_bin_cols": [],
    }
    with pytest.raises(ValueError, match="Observation columns missing"):
        SchemaValidator(schema_missing_obs)

    schema_missing_actions = {
        "observation_cols": ["x"],
        "action_cols": [NO_MEDICATION_ACTION],
        "model_input_cols": ["x", DELTA_COLUMN],
        "y_cont_cols": [],
        "y_bin_cols": [],
    }
    with pytest.raises(ValueError, match="Action columns missing"):
        SchemaValidator(schema_missing_actions)


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

    # Non-finite categorical values should be sanitized before argmax
    sample_nf = pd.Series({'c1': 0.0, 'cat_a': float('nan'), 'cat_b': float('nan')})
    enforced_nf = validator.enforce_categorical_groups(sample_nf.copy())
    assert float(enforced_nf[['cat_a', 'cat_b']].sum()) == pytest.approx(1.0, abs=1e-6)


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

    negative_weights = {
        'observation_cols': observation_cols,
        'distributions': {
            'all': {
                'clusters': [{'mean': np.array([0.0, 0.0]), 'cov': np.eye(2), 'weight': -1.0}],
                'weights': np.array([1.0]),
                'power_transformer': IdentityTransformer(),
                'num_samples': 1,
            }
        }
    }
    with pytest.raises(ValueError, match="negative mixture weights"):
        InitialStateSampler(
            observation_cols=observation_cols,
            y_categorical_groups=y_groups,
            variable_bounds=bounds,
            initial_state_payload=negative_weights,
        )

    zero_weights = {
        'observation_cols': observation_cols,
        'distributions': {
            'all': {
                'clusters': [{'mean': np.array([0.0, 0.0]), 'cov': np.eye(2), 'weight': 0.0}],
                'weights': np.array([1.0]),
                'power_transformer': IdentityTransformer(),
                'num_samples': 1,
            }
        }
    }
    with pytest.raises(ValueError, match="weights sum to zero"):
        InitialStateSampler(
            observation_cols=observation_cols,
            y_categorical_groups=y_groups,
            variable_bounds=bounds,
            initial_state_payload=zero_weights,
        )

    # sample() error paths (post-init)
    sampler_empty = InitialStateSampler(
        observation_cols=observation_cols,
        y_categorical_groups=y_groups,
        variable_bounds=bounds,
        initial_state_payload=valid_gaussians,
    )
    sampler_empty.initial_state_gaussians = None
    with pytest.raises(ValueError, match="not found or invalid"):
        sampler_empty.sample('all')

    sampler_bad = InitialStateSampler(
        observation_cols=observation_cols,
        y_categorical_groups=y_groups,
        variable_bounds=bounds,
        initial_state_payload=valid_gaussians,
    )
    with pytest.raises(ValueError, match="not found"):
        sampler_bad.sample('missing_cohort')

    # weights length mismatch
    sampler_bad.initial_state_gaussians['all']['weights'] = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="mixture weights"):
        sampler_bad.sample('all')

    # missing transformer
    sampler_bad.initial_state_gaussians['all']['weights'] = np.array([1.0])
    sampler_bad.initial_state_gaussians['all']['power_transformer'] = None
    with pytest.raises(ValueError, match="PowerTransformer"):
        sampler_bad.sample('all')


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

    # reliability=1.0 triggers s_diff fallback (should remain finite)
    calc = make_reward_calc(reliability=1.0)
    obs_cols = ['metric', 'other']
    reward = calc.calculate_reward(prev, nxt, obs_cols, bounds_check=lambda s: (True, []))
    assert np.isfinite(reward)


def test_action_validator_structure_errors():
    cols = ["a", NO_MEDICATION_ACTION]
    v = ActionValidator(action_cols=cols, no_med_idx=1)

    with pytest.raises(ValueError, match="numpy array"):
        v.validate_action_structure([0, 1])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-finite"):
        v.validate_action_structure(np.array([0, np.nan]))

    with pytest.raises(ValueError, match="binary"):
        v.validate_action_structure(np.array([0, 0.5]))


def test_manage_state_scaling_error_paths():
    scaler = SimpleNamespace(
        feature_names_in_=np.array(["a"], dtype=object),
        mean_=np.array([0.0]),
        scale_=np.array([1.0]),
        var_=np.array([1.0]),
        with_mean=True,
        with_std=True,
    )
    with pytest.raises(TypeError, match="pandas DataFrame"):
        manage_state_scaling([1, 2, 3], scaler)  # type: ignore[arg-type]

    df = pd.DataFrame({"b": [1.0]})
    with pytest.raises(ValueError, match="not found in scaler feature names"):
        manage_state_scaling(df, scaler)


def test_manage_state_scaling_no_feature_names_is_noop():
    df = pd.DataFrame({"a": [1.0]})
    out = manage_state_scaling(df, scaler=SimpleNamespace(), normalize=True)
    pd.testing.assert_frame_equal(out, df)


def test_manage_state_scaling_inverse_and_flags():
    df = pd.DataFrame({"a": [2.0]})
    scaler = SimpleNamespace(
        feature_names_in_=np.array(["a"], dtype=object),
        mean_=np.array([1.0]),
        scale_=np.array([2.0]),
        var_=np.array([1.0]),
        with_mean=False,
        with_std=False,
    )
    # with_mean/with_std False => identity in both directions
    out = manage_state_scaling(df, scaler=scaler, normalize=True)
    pd.testing.assert_frame_equal(out, df.astype(np.float32))
    out2 = manage_state_scaling(df, scaler=scaler, normalize=False)
    pd.testing.assert_frame_equal(out2, df.astype(np.float32))

    # scale=0 should avoid division-by-zero
    scaler2 = SimpleNamespace(
        feature_names_in_=np.array(["a"], dtype=object),
        mean_=np.array([1.0]),
        scale_=np.array([0.0]),
        var_=np.array([1.0]),
        with_mean=True,
        with_std=True,
    )
    out3 = manage_state_scaling(df, scaler=scaler2, normalize=True)
    assert np.isfinite(out3["a"].iloc[0])
