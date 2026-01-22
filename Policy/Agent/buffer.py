from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Rollout:
    obs: np.ndarray
    actions: np.ndarray
    logp_old: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    lam: float,
):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + values
    return adv.astype(np.float32), ret.astype(np.float32)
