import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from ALPACA_DT_Sim.alpaca_env import ALPACA_DT_SimEnv
from ALPACA_DT_Sim.artifact_loader import ArtifactLoader


@pytest.fixture(scope="module")
def alpaca_dir() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="module")
def artifact_loader(alpaca_dir: Path) -> ArtifactLoader:
    """Provide an artifact loader rooted at the ALPACA package."""
    return ArtifactLoader(artifact_root=alpaca_dir)


@pytest.fixture(scope="module")
def env(artifact_loader: ArtifactLoader):
    # Use the local ALPACA artifacts (scalers, model, schema, bounds) with Gaussian starts
    return ALPACAEnv(cohort_type='all', artifact_loader=artifact_loader)


def test_schema_and_model_inputs_match_scalers(env: ALPACAEnv):
    """Round-trip scale/unscale intersecting model inputs to ensure schema and scalers align."""
    # model_input_cols should include obs + actions + next_visit_months (order defined by schema)
    assert isinstance(env.model_input_cols, list) and len(env.model_input_cols) > 0

    # Scaler-X columns must be a subset of model_input_cols and transform should be consistent
    sx_cols = list(getattr(env.scaler_X, 'feature_names_in_', []))
    assert set(sx_cols).issubset(set(env.model_input_cols))

    # Build a model input row like env.step constructs, test round-trip scaling on intersecting cols
    obs, _ = env.reset()
    obs_s = pd.Series(obs, index=env.observation_cols)
    act = np.zeros(len(env.action_cols), dtype=float)
    act_s = pd.Series(act, index=env.action_cols)
    row = pd.concat([obs_s, act_s])
    row[env.delta_col] = env.time_delta_val

    df = pd.DataFrame([row])[env.model_input_cols]
    cols = [c for c in sx_cols if c in df.columns]
    if cols:
        before = df[cols].copy()
        scaled = env.manage_state_scaling(df[cols], env.scaler_X, normalize=True)
        unscaled = env.manage_state_scaling(scaled, env.scaler_X, normalize=False)
        # Round-trip should match original within numerical tolerances
        np.testing.assert_allclose(unscaled.values, before.values, rtol=1e-5, atol=1e-5)


def test_scaler_alignment_invariants(env: ALPACAEnv):
    """Validate scaler metadata has no duplicates, matches schema, and stats lengths align."""
    sx_names = list(env.scaler_X.feature_names_in_)
    sy_names = list(env.scaler_y.feature_names_in_)

    # No duplicates and no extras relative to model inputs
    assert len(sx_names) == len(set(sx_names))
    assert len(sy_names) == len(set(sy_names))
    assert not set(sx_names) - set(env.model_input_cols)

    # All continuous inputs and delta must be scaled
    expected_inputs = set(env.cont_obs_cols + [env.delta_col])
    assert expected_inputs.issubset(set(sx_names))

    # All continuous outputs must be present in scaler_y
    assert set(env.model_cont_output_cols).issubset(set(sy_names))
    # Stats arrays should match feature name lengths
    assert len(env.scaler_X.mean_) == len(sx_names)
    assert len(env.scaler_y.mean_) == len(sy_names)


def test_spaces_and_bounds(env: ALPACAEnv):
    """Ensure observation/action spaces match schema lengths and have valid bounds."""
    # Action/observation shape checks
    assert env.action_space.n == len(env.action_cols)
    assert env.observation_space.shape[0] == len(env.observation_cols)
    assert np.all(env.observation_space.low < env.observation_space.high)


def test_action_constraints(env: ALPACAEnv):
    """Confirm invalid actions (none or NoMed+other) terminate with -10 reward."""
    env.reset()
    # No action -> done with strong negative reward
    no_action = np.zeros(len(env.action_cols), dtype=int)
    _, reward, done, _, info = env.step(no_action)
    assert done and reward == -10.0

    # No Medication concurrent with others -> done -10 (if present)
    env.reset()
    if 'No Medication_active' in env.action_cols:
        a = np.zeros(len(env.action_cols), dtype=int)
        idx = env.action_cols.index('No Medication_active')
        a[idx] = 1
        a[(idx + 1) % len(env.action_cols)] = 1
        _, reward, done, _, info = env.step(a)
        assert done and reward == -10.0


def test_subject_age_progression(env: ALPACAEnv):
    """Verify subject_age advances by the configured time_delta each step."""
    obs0, _ = env.reset()
    age0 = pd.Series(obs0, index=env.observation_cols).get('subject_age', None)
    a = np.zeros(len(env.action_cols), dtype=int)
    obs1, _, _, _, _ = env.step(a + (np.eye(len(env.action_cols), dtype=int)[0]))
    age1 = pd.Series(obs1, index=env.observation_cols).get('subject_age', None)
    if age0 is not None and age1 is not None:
        assert pytest.approx(age0 + env.time_delta_val / 12.0, rel=1e-6, abs=1e-6) == age1


def test_reward_scaling_and_clipping(env: ALPACAEnv):
    """Check reliable-change reward is scaled and clipped symmetrically for +/- deltas."""
    # Use a valid base observation, adjust metric by +/- up to 3 within bounds
    metric = env.reward_metric
    assert metric in env.observation_cols
    prev, _ = env.reset()
    prev = prev.copy()
    i = env.observation_cols.index(metric)
    base = float(prev[i])

    # Determine safe positive/negative deltas within bounds
    lb = -np.inf
    ub = np.inf
    if getattr(env, 'variable_bounds', None) is not None and metric in env.variable_bounds.index:
        lb = float(env.variable_bounds.loc[metric, 'lower_bound'])
        ub = float(env.variable_bounds.loc[metric, 'upper_bound'])

    r_xx = float(np.clip(getattr(env, 'reliability_rxx', 0.9), 0.0, 0.999999))
    s_diff = float(np.sqrt(2.0 * (1.0 - r_xx)) * 1.0)

    # Positive direction
    max_pos = min(3.0, ub - base)
    if max_pos > 1e-6:
        nxt = prev.copy()
        nxt[i] = base + max_pos
        r_pos = env.calculate_reward(prev, nxt)
        expected = float(np.clip(10.0 * (max_pos / s_diff), -10.0, 10.0))
        assert pytest.approx(expected, rel=1e-6, abs=1e-6) == r_pos

    # Negative direction
    max_neg = min(3.0, base - lb)
    if max_neg > 1e-6:
        nxt = prev.copy()
        nxt[i] = base - max_neg
        r_neg = env.calculate_reward(prev, nxt)
        expected = float(np.clip(-10.0 * (max_neg / s_diff), -10.0, 10.0))
        assert pytest.approx(expected, rel=1e-6, abs=1e-6) == r_neg


def test_categorical_one_hot_groups(env: ALPACAEnv):
    """After a step, categorical groups must remain one-hot."""
    # After one step, categorical groups should remain one-hot (sum to 1) where applicable
    obs, _ = env.reset()
    a = env.action_space.sample()
    obs2, _, done, _, _ = env.step(a)
    s = pd.Series(obs2, index=env.observation_cols)
    for _, cols in getattr(env, 'y_categorical_groups', {}).items():
        cols = [c for c in cols if c in s.index]
        if not cols:
            continue
        total = float(s[cols].sum())
        # Allow occasional absence if schema columns missing; otherwise expect one-hot
        assert pytest.approx(total, abs=1e-6) == 1.0


def test_model_input_order_matches_schema(env: ALPACAEnv, alpaca_dir: Path):
    """Verify model_input_cols reported by env exactly match schema ordering."""
    # Ensure the env uses the same order as columns_schema.json model_input_cols
    import json
    schema_path = alpaca_dir / 'columns_schema.json'
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    schema_cols = schema.get('model_input_cols', [])
    assert env.model_input_cols == schema_cols


def test_env_requires_gaussian_artifact(alpaca_dir: Path, tmp_path):
    """Environment initialization should fail when the Gaussian artifact is missing."""
    missing_artifact = tmp_path / 'missing_initial_state.joblib'
    with pytest.raises(ValueError):
        missing_loader = ArtifactLoader(
            artifact_root=alpaca_dir,
            initial_state_gaussian_path=missing_artifact,
        )
        ALPACAEnv(cohort_type='all', artifact_loader=missing_loader)


def test_gaussian_initial_state_sampling(artifact_loader: ArtifactLoader):
    """Validate Gaussian artifact loads, samples within bounds, and preserves one-hot groups."""
    env_gauss = ALPACAEnv(cohort_type='all', artifact_loader=artifact_loader)
    assert env_gauss.initial_state_gaussians is not None
    state, _ = env_gauss.reset(seed=123)
    assert state.shape == (len(env_gauss.observation_cols),)
    valid, violations = env_gauss.check_state_bounds(state)
    assert valid, f"Sampled state violated bounds: {violations}"
    series = pd.Series(state, index=env_gauss.observation_cols)
    for _, cols in env_gauss.y_categorical_groups.items():
        cols = [c for c in cols if c in series.index]
        if not cols:
            continue
        total = float(series[cols].sum())
        assert pytest.approx(total, abs=1e-6) == 1.0
    
    # Load artifact from loader path for validation
    path = artifact_loader.paths.initial_state_gaussians_path
    artifact_payload = joblib.load(path)
    
    assert artifact_payload['observation_cols'] == env_gauss.observation_cols
    cohort_stats = artifact_payload['distributions'][env_gauss.cohort_type]
    assert len(cohort_stats['clusters']) > 0


def test_env_rejects_missing_scaled_inputs(
    alpaca_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_loader: ArtifactLoader,
):
    """scaler_X missing required features should raise during env construction."""
    real_load = joblib.load

    def fake_load(path, *args, **kwargs):
        obj = real_load(path, *args, **kwargs)
        if str(path).endswith('scaler_X.joblib'):
            names = list(obj.feature_names_in_)
            if 'next_visit_months' in names:
                idx = names.index('next_visit_months')
                obj.feature_names_in_ = np.array([n for i, n in enumerate(names) if i != idx], dtype=object)
                obj.n_features_in_ = len(obj.feature_names_in_)
                if hasattr(obj, 'mean_'):
                    obj.mean_ = np.delete(obj.mean_, idx)
                if hasattr(obj, 'scale_'):
                    obj.scale_ = np.delete(obj.scale_, idx)
                if hasattr(obj, 'var_'):
                    obj.var_ = np.delete(obj.var_, idx)
        return obj

    monkeypatch.setattr(joblib, "load", fake_load)
    with pytest.raises(ValueError, match="scaler_X.*missing required continuous inputs"):
        ALPACAEnv(artifact_loader=artifact_loader)


def test_env_rejects_missing_output_scaler(
    alpaca_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_loader: ArtifactLoader,
):
    """scaler_y missing outputs should raise during env construction."""
    real_load = joblib.load

    def fake_load(path, *args, **kwargs):
        obj = real_load(path, *args, **kwargs)
        if str(path).endswith('scaler_y.joblib'):
            names = list(obj.feature_names_in_)
            if 'ADNI_MEM' in names:
                idx = names.index('ADNI_MEM')
                obj.feature_names_in_ = np.array([n for i, n in enumerate(names) if i != idx], dtype=object)
                obj.n_features_in_ = len(obj.feature_names_in_)
                if hasattr(obj, 'mean_'):
                    obj.mean_ = np.delete(obj.mean_, idx)
                if hasattr(obj, 'scale_'):
                    obj.scale_ = np.delete(obj.scale_, idx)
                if hasattr(obj, 'var_'):
                    obj.var_ = np.delete(obj.var_, idx)
        return obj

    monkeypatch.setattr(joblib, "load", fake_load)
    with pytest.raises(ValueError, match="scaler_y.*missing required model continuous outputs"):
        ALPACAEnv(artifact_loader=artifact_loader)


def test_sequence_autoregression_and_rollout(env: ALPACAEnv):
    """Roll out several steps ensuring sequence length increments and outputs stay finite."""
    # Short rollout to ensure sequence grows and step returns valid outputs
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    steps = 4
    prev_len = 0
    for t in range(steps):
        a = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(a)
        assert isinstance(obs, np.ndarray)
        assert np.isfinite(reward)
        assert isinstance(done, bool) and isinstance(truncated, bool)
        # sequence_length should increase by 1 each step
        assert info.get('sequence_length', 0) == prev_len + 1
        prev_len = info.get('sequence_length', prev_len)
        if done or truncated:
            break


def test_bounds_checker_logic(env: ALPACAEnv):
    """Bounds checker should pass valid states and report detailed violations for invalid ones."""
    # 1. A valid state should pass
    state, _ = env.reset()
    is_valid, violations = env.check_state_bounds(state)
    assert is_valid
    assert len(violations) == 0

    # 2. An invalid state should fail with details
    state_invalid = state.copy()
    # Pick a variable with known bounds to violate
    metric_to_violate = 'ADNI_MEM'  # This is usually well-defined
    if metric_to_violate not in env.observation_cols or env.variable_bounds is None or metric_to_violate not in env.variable_bounds.index:
        pytest.skip(f"Cannot test bounds violation; '{metric_to_violate}' has no defined bounds.")

    metric_idx = env.observation_cols.index(metric_to_violate)
    upper_bound = env.variable_bounds.loc[metric_to_violate, 'upper_bound']

    # Violate upper bound
    state_invalid[metric_idx] = upper_bound + 1.0
    is_valid, violations = env.check_state_bounds(state_invalid)
    assert not is_valid
    assert len(violations) == 1
    assert violations[0]['variable'] == metric_to_violate
    assert pytest.approx(upper_bound + 1.0, rel=1e-6, abs=1e-6) == violations[0]['value']

    # Violate lower bound
    lower_bound = env.variable_bounds.loc[metric_to_violate, 'lower_bound']
    state_invalid[metric_idx] = lower_bound - 1.0
    is_valid, violations = env.check_state_bounds(state_invalid)
    assert not is_valid
    assert len(violations) == 1
    assert violations[0]['variable'] == metric_to_violate
    assert pytest.approx(lower_bound - 1.0, rel=1e-6, abs=1e-6) == violations[0]['value']


def test_step_terminates_on_out_of_bounds_next_state(env: ALPACAEnv):
    """
    Tests that the episode terminates if the model predicts a next state
    that is outside the defined variable bounds.
    """
    env.reset(seed=42)
    
    # Find a metric with a defined upper bound to violate
    metric_to_violate = 'ADNI_MEM'
    if metric_to_violate not in env.observation_cols or env.variable_bounds is None or metric_to_violate not in env.variable_bounds.index:
        pytest.skip(f"Cannot test bounds violation; '{metric_to_violate}' has no defined bounds.")

    upper_bound = env.variable_bounds.loc[metric_to_violate, 'upper_bound']

    # The model outputs scaled values. We need to produce a scaled value that,
    # when unscaled, will exceed the bound.
    # y_unscaled = y_scaled * scale + mean  => y_scaled = (y_unscaled - mean) / scale
    metric_idx_y = env.model_cont_output_cols.index(metric_to_violate)
    scaler_y_idx = list(env.scaler_y.feature_names_in_).index(metric_to_violate)
    mean = env.scaler_y.mean_[scaler_y_idx]
    scale = env.scaler_y.scale_[scaler_y_idx]
    
    # Target a value just above the upper bound
    violation_value_unscaled = upper_bound + 1.0
    violation_value_scaled = (violation_value_unscaled - mean) / scale

    # Mock the model's output
    mock_pred_cont = torch.zeros((1, 1, len(env.model_cont_output_cols)), device=env.device)
    mock_pred_cont[0, 0, metric_idx_y] = violation_value_scaled
    
    mock_pred_bin = torch.zeros((1, 1, len(env.model_binary_output_cols)), device=env.device)

    with patch.object(env.model, 'forward', return_value=(mock_pred_cont, mock_pred_bin)):
        action = np.zeros(len(env.action_cols), dtype=int)
        for idx, name in enumerate(env.action_cols):
            if name != 'No Medication_active':
                action[idx] = 1
                break
        # The state before the step
        state_before = env.state.copy()
        next_state, reward, done, truncated, info = env.step(action)

    # The episode should be 'done'
    assert done, "Episode should terminate on out-of-bounds state."
    # Truncated should be False, as this is a terminal condition
    assert not truncated
    # Reward should be 0.0 for this type of termination
    assert reward == 0.0
    # Info should contain the reason
    assert info.get('termination_reason') == 'state_out_of_bounds'
    assert len(info.get('out_of_bounds_variables', [])) == 1
    violation_info = info['out_of_bounds_variables'][0]
    assert violation_info['variable'] == metric_to_violate
    assert np.isclose(violation_info['value'], violation_value_unscaled)
    
    # The state should not have been updated to the invalid state
    np.testing.assert_allclose(next_state, state_before, rtol=1e-6)


def test_step_rejects_non_finite_inputs(alpaca_dir: Path, artifact_loader: ArtifactLoader):
    """Non-finite state values should trigger a ValueError before stepping."""
    env_local = ALPACAEnv(cohort_type='all', artifact_loader=artifact_loader)
    env_local.state = np.full_like(env_local.state, np.nan)
    valid_action = np.zeros(len(env_local.action_cols), dtype=int)
    action_idx = 0 if env_local._no_med_idx != 0 else 1
    valid_action[action_idx] = 1

    with pytest.raises(ValueError, match="non-finite"):
        env_local.step(valid_action)


def test_episode_truncation_vs_termination(env: ALPACAEnv):
    """Distinguish truncated episodes at max length from true terminations."""
    env.reset()
    steps = env.max_episode_length - 1
    valid_action = np.zeros(len(env.action_cols), dtype=int)
    valid_action[0 if env._no_med_idx != 0 else 1] = 1
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(valid_action)
        if terminated or truncated:
            pytest.fail("Episode ended earlier than expected.")

    # Final step should truncate but not terminate
    _, _, terminated, truncated, _ = env.step(valid_action)
    assert not terminated
    assert truncated


def test_constraint_violation_mid_episode(env: ALPACAEnv):
    """Constraint violations after an initial valid step should immediately terminate with -10."""
    env.reset()
    valid_action = np.zeros(len(env.action_cols), dtype=int)
    valid_action[0 if env._no_med_idx != 0 else 1] = 1
    env.step(valid_action)

    if 'No Medication_active' in env.action_cols:
        idx_no_med = env.action_cols.index('No Medication_active')
        invalid_action = np.zeros(len(env.action_cols), dtype=int)
        invalid_action[idx_no_med] = 1
        invalid_action[(idx_no_med + 1) % len(env.action_cols)] = 1
        _, reward, terminated, truncated, info = env.step(invalid_action)
        assert terminated
        assert not truncated
        assert reward == -10.0
        assert info.get('constraint_violation') == 'no_medication_with_other_actions'


def test_categorical_one_hot_stability(env: ALPACAEnv):
    """Categorical one-hot groups should remain valid across a short rollout."""
    obs, _ = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        s = pd.Series(obs, index=env.observation_cols)
        for _, cols in env.y_categorical_groups.items():
            cols = [c for c in cols if c in s.index]
            if not cols:
                continue
            total = float(s[cols].sum())
            assert pytest.approx(1.0, abs=1e-6) == total
        if terminated or truncated:
            break


def test_mc_dropout_uncertainty_fields(alpaca_dir: Path, artifact_loader: ArtifactLoader):
    """With mc_samples>0, info must expose uncertainty metrics and CIs for the reward metric."""
    env_local = ALPACAEnv(mc_samples=3, artifact_loader=artifact_loader)
    env_local.reset()
    action = np.zeros(len(env_local.action_cols), dtype=int)
    action[0 if env_local._no_med_idx != 0 else 1] = 1
    _, _, terminated, truncated, info = env_local.step(action)
    assert not terminated and not truncated
    assert 'mean_cont_uncertainty' in info
    metric_key = f"{env_local.reward_metric.lower()}_ci_lower"
    assert metric_key in info
    assert np.isfinite(info['mean_cont_uncertainty'])
    env_local.close()


def test_reset_seed_determinism(env: ALPACAEnv):
    """Resetting with the same seed should reproduce identical initial states."""
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    obs3, _ = env.reset(seed=124)
    np.testing.assert_allclose(obs1, obs2, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(obs1, obs3, rtol=0, atol=0)


def test_invalid_action_shape_raises(alpaca_dir: Path, artifact_loader: ArtifactLoader):
    """Actions with wrong shape should raise instead of being coerced."""
    env_local = ALPACAEnv(cohort_type='all', artifact_loader=artifact_loader)
    env_local.reset()
    wrong_action = np.zeros(len(env_local.action_cols) + 1, dtype=int)
    with pytest.raises((AssertionError, ValueError, IndexError)):
        env_local.step(wrong_action)
    env_local.close()


def test_model_nan_outputs_raise(monkeypatch: pytest.MonkeyPatch, alpaca_dir: Path, artifact_loader: ArtifactLoader):
    """NaN continuous model outputs should raise before corrupting state."""
    env_local = ALPACAEnv(cohort_type='all', artifact_loader=artifact_loader)
    env_local.reset()

    def fake_forward(*args, **kwargs):
        cont = torch.full((1, 1, len(env_local.model_cont_output_cols)), float('nan'), device=env_local.device)
        bin_logits = torch.zeros((1, 1, len(env_local.model_binary_output_cols)), device=env_local.device)
        return cont, bin_logits

    monkeypatch.setattr(env_local.model, "forward", fake_forward)
    action = np.zeros(len(env_local.action_cols), dtype=int)
    action[0 if env_local._no_med_idx != 0 else 1] = 1
    with pytest.raises(ValueError):
        env_local.step(action)
    env_local.close()


def test_missing_delta_fails_fast(monkeypatch: pytest.MonkeyPatch, alpaca_dir: Path, artifact_loader: ArtifactLoader):
    """Dropping the time-delta column from model inputs should raise a clear error."""
    env_local = ALPACAEnv(cohort_type='all', artifact_loader=artifact_loader)
    env_local.reset()

    real_build = env_local._build_model_input

    def fake_build(obs, action):
        df = real_build(obs, action)
        return df.drop(columns=[env_local.delta_col])

    monkeypatch.setattr(env_local, "_build_model_input", fake_build)
    action = np.zeros(len(env_local.action_cols), dtype=int)
    action[0 if env_local._no_med_idx != 0 else 1] = 1
    with pytest.raises(ValueError, match="missing required columns"):
        env_local.step(action)
    env_local.close()


def test_info_payload_and_spaces(env: ALPACAEnv):
    """Observation/action spaces must contain emitted states; info should track sequence_length."""
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert info == {}

    action = env.action_space.sample()
    obs, _, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert isinstance(info.get('sequence_length'), int)
    if env.mc_samples <= 0:
        assert 'mean_cont_uncertainty' not in info
