"""
ALPACA environment package for autoregressive RL.

This makes the folder importable so we can use relative imports
for the local MoE Transformer implementation.
"""

from .alpaca_env import ALPACAEnv

__all__ = ["ALPACAEnv"]
