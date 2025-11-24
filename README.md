# ALPACA: Alzheimer’s Learning Platform for Agent-based Care Advancement

<p align="center">
  <img src="PATH_TO_IMAGE/alpaca.png" width="250" />
</p>

## Overview

ALPACA is a Gymnasium-compatible Reinforcement Learning (RL) environment designed to simulate and optimize treatment strategies for Alzheimer's Disease (AD). It uses a pre-trained dynamics model (a Mixture of Experts Transformer) to forecast patient progression based on their current clinical state and prescribed medications.

The objective is to train RL agents to prescribe optimal combinations of FDA-approved medications to maximize patient cognitive outcomes (e.g., stabilizing or improving memory scores) over time.

## Installation

(Assuming this is installed via pip)

```bash
pip install alpaca-rl
```

*Note: Ensure you have the required data artifacts (models, scalers, schema) in your working directory or specified data path.*

## Quick Start

Here is a minimal example of how to initialize the environment and run a simple loop.

```python
import gymnasium as gym
import numpy as np
from alpaca_env import ALPACAEnv  # Adjust import based on final package structure

# 1. Initialize the environment
env = ALPACAEnv(
    reward_metric='ADNI_MEM',  # Optimize for ADNI Memory Score
    cohort_type='all'          # Sample from all patient types
)

# 2. Reset to get the initial patient state
obs, info = env.reset()
print("Initial Patient State:", obs[:5], "...") 

# 3. Interaction Loop
done = False
truncated = False
total_reward = 0

while not (done or truncated):
    # Random action: Select a subset of medications (MultiBinary)
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Episode finished with total reward: {total_reward}")
env.close()
```

## Environment Details

### Observation Space
The observation space is a `Box` (continuous) representing the patient's clinical state. It consists of **21 features** in total, including cognitive scores, brain imaging biomarkers, and demographics.

**Continuous Features (12):**
- **Cognitive & Functional Scores**: `ADNI_MEM` (Memory), `ADNI_EF2` (Executive Function).
- **Demographics**: `subject_age`.
- **Biomarkers (CSF/Imaging)**: `TAU_data`, `ABETA`.
- **Brain Volumetrics**: `Ventricles`, `Hippocampus`, `WholeBrain`, `Entorhinal`, `Fusiform`, `MidTemp`, `ICV`.

**Categorical/Binary Features (9):**
- **Gender**: `PTGENDER_Female`, `PTGENDER_Male`.
- **Race**: `PTRACCAT_Am Indian/Alaskan`, `PTRACCAT_Asian`, `PTRACCAT_Black`, `PTRACCAT_Hawaiian/Other PI`, `PTRACCAT_More than one`, `PTRACCAT_Unknown`, `PTRACCAT_White`.

*Note: The bounds for continuous variables are strictly enforced based on `ADNI_Variable_Bounds.csv`. If a patient state evolves outside these bounds, the episode terminates.*

### Action Space
The action space is `MultiBinary(17)`, representing the prescription status of various medication classes.
- `1`: Medication Active
- `0`: Medication Inactive

**Available Medications:**
1. `AD Treatment_active`
2. `Alpha Blocker_active`
3. `Analgesic_active`
4. `Antidepressant_active`
5. `Antihypertensive_active`
6. `Bone Health_active`
7. `Diabetes Medication_active`
8. `Diuretic_active`
9. `NSAID_active`
10. `No Medication_active` (Baseline)
11. `Other_active`
12. `PPI_active`
13. `SSRI_active`
14. `Statin_active`
15. `Steroid_active`
16. `Supplement_active`
17. `Thyroid Hormone_active`

*Constraints*: The environment validates actions to ensure consistency (e.g., checking mutually exclusive combinations if defined).

### Reward Function
The reward is designed to incentivize **reliable improvement** or **stabilization** in the target metric (default: `ADNI_MEM`) while filtering out measurement noise.

It uses a **Reliable Change Index (RCI)** formulation:

$$ R_t = \text{clip}\left( 10 \times \frac{\Delta}{S_{\text{diff}}}, -10, 10 \right) $$

Where:
- **$\Delta$**: Change in the metric from step $t$ to $t+1$ (e.g., $\text{ADNI\_MEM}_{t+1} - \text{ADNI\_MEM}_t$).
- **$S_{\text{diff}}$**: The standard difference, calculated as $\sqrt{2(1 - r_{xx})} \times SD$.
    - $r_{xx}$: Test-retest reliability coefficient (default $\approx 0.9$).
    - $SD$: Standard deviation of the metric (assumed $1.0$ for z-scored ADNI metrics).

**Key Behaviors:**
- **Positive Reward**: Significant improvement relative to noise.
- **Negative Reward**: Significant decline.
- **Zero Reward**: If the resulting state violates physiological bounds (episode terminates).
- **Clipping**: Rewards are clipped between -10 and +10 to stabilize training.

### Dynamics Model
The environment is powered by a **Mixture-of-Experts (MoE) Transformer** trained on the ADNI dataset. It predicts the patient's state at the next visit (e.g., 6 months later) given the current state and chosen actions.
- **Uncertainty**: You can enable Monte Carlo (MC) Dropout sampling (`mc_samples > 0` in init) to get uncertainty estimates in the `info` dictionary, useful for safe RL or exploration.

## Configuration

The `ALPACAEnv` constructor accepts several parameters to customize the simulation:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_delta_months` | `float` | `6.0` | The simulated time duration between steps (visits). |
| `reward_metric` | `str` | `'ADNI_MEM'` | The clinical feature to optimize (e.g., 'ADNI_MEM', 'CDRSB'). |
| `cohort_type` | `str` | `'all'` | Subset of patients to sample initial states from: `'all'`, `'impaired'`, or `'healthy'`. |
| `mc_samples` | `int` | `0` | Number of MC dropout samples for uncertainty estimation (0 to disable). |

## Included Artifacts
The package comes bundled with the necessary artifacts:
1. `best_moe_transformer_model.pt`: The trained dynamics model.
2. `scaler_X.joblib` & `scaler_y.joblib`: Data scalers for input/output.
3. `columns_schema.json`: Defines the feature order and types.
4. `ADNI_Variable_Bounds.csv`: Min/max bounds for validation.
5. `initial_state_gaussians.joblib`: Parameters for sampling new patients.

## RL Library Integration

ALPACA is fully compatible with **Stable Baselines 3 (SB3)**.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
vec_env = make_vec_env(lambda: ALPACAEnv(), n_envs=4)

# Train an agent
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)

# Save and evaluate
model.save("ppo_alpaca_agent")
```

## Disclaimer
This package is for research purposes only and is not intended for clinical use. Please use with caution and consult a medical professional before using any of the policies or models generated by training on this environment.
