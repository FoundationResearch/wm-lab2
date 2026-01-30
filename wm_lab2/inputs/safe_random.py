from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class SafeRandomPolicy(Policy):
    """
    A safer "random" policy for MineDojo MultiDiscrete actions.

    Key idea: start from env.action_space.no_op() and only randomize a small
    subset of dimensions corresponding to locomotion + camera. This avoids
    invalid inventory/equip actions like "equip air" that strict wrappers reject.

    Defaults assume a MineDojo-style layout where early dims are movement/camera:
      - idx 0: forward/backward (3)
      - idx 1: left/right (3)
      - idx 2: jump/sneak/sprint-ish (varies)
      - idx 3-4: camera (varies, often 25/25)
    For unknown envs, you can override `active_dims`.
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator
    active_dims: Sequence[int] = (0, 1, 2, 3, 4)
    non_stationary_prob: float = 0.95

    def reset(self, obs: Dict[str, Any]) -> None:
        # Stateless by default
        return None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        a = np.array(self.noop, copy=True)

        dims: Iterable[int] = (d for d in self.active_dims if 0 <= int(d) < len(a))
        for d in dims:
            n = int(self.nvec[d])
            if n <= 1:
                continue
            a[d] = int(self.rng.integers(0, n))

        # Avoid all-zero action (standing still) by forcing one active dim non-zero.
        if self.rng.random() < self.non_stationary_prob and np.all(a == self.noop):
            cand = [int(d) for d in self.active_dims if 0 <= int(d) < len(a) and int(self.nvec[int(d)]) > 1]
            if cand:
                d = int(self.rng.choice(cand))
                n = int(self.nvec[d])
                a[d] = int(self.rng.integers(1, n))

        return a.astype(np.int64, copy=False)


