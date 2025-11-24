"""Reliable-change reward computation for ALPACA."""

from typing import Callable, Tuple

import numpy as np
import pandas as pd


class RewardCalculator:
    """Compute reliable-change rewards with bound-awareness."""

    def __init__(self, reward_metric: str, reliability_rxx: float):
        self.reward_metric = reward_metric
        self.reliability_rxx = float(reliability_rxx)

    def calculate_reward(
        self,
        prev_obs: Tuple[float, ...],
        next_obs: Tuple[float, ...],
        observation_cols,
        bounds_check: Callable[[object], Tuple[bool, object]],
    ) -> float:
        """
        Reliable-change reward (bounded to [-10, 10]):

            r_t = clip( 10 * Δ / S_diff, -10, 10 )

        where Δ = S_{t+1} - S_t for the selected reward_metric, and
            S_diff = sqrt(2 * (1 - r_xx)) * SD.

        Assumptions:
        - ADNI_MEM is z-scaled ⇒ SD = 1.
        - Test–retest reliability r_xx ≈ 0.9 (can override via self.reliability_rxx).

        If the next state violates bounds, reward is neutralized to 0.
        """
        prev_series = pd.Series(prev_obs, index=observation_cols)
        next_series = pd.Series(next_obs, index=observation_cols)
        metric = self.reward_metric
        if metric not in prev_series.index:
            return 0.0

        delta = float(next_series[metric] - prev_series[metric])

        r_xx = float(np.clip(self.reliability_rxx, 0.0, 0.999999))
        sd = 1.0
        s_diff = float(np.sqrt(2.0 * (1.0 - r_xx)) * sd)
        if not np.isfinite(s_diff) or s_diff <= 1e-12:
            s_diff = 0.4472135955

        scaled = 10.0 * (delta / s_diff)
        base_reward = float(np.clip(scaled, -10.0, 10.0))

        within_bounds, _ = bounds_check(next_obs)
        if not within_bounds:
            base_reward = 0.0

        return base_reward
