from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class MyPolicy:
    """
    Minimal custom policy template.

    Requirements:
    - reset(obs) at episode start
    - act(obs) -> np.ndarray action of shape (D,)
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator

    def reset(self, obs: Dict[str, Any]) -> None:
        # optional: clear hidden states, etc.
        return None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # IMPORTANT: start from noop so you don't accidentally trigger inventory/equip actions.
        a = np.array(self.noop, copy=True)

        # Example: simple forward walking with small random camera jitter (dims 0,3,4 are common).
        if len(a) > 0:
            a[0] = 1  # forward (MineDojo common convention)
        if len(a) > 4:
            a[3] = int(self.rng.integers(0, int(self.nvec[3])))
            a[4] = int(self.rng.integers(0, int(self.nvec[4])))

        return a.astype(np.int64, copy=False)


def make_policy(nvec: np.ndarray, noop: np.ndarray, rng: np.random.Generator) -> MyPolicy:
    """
    Entry point used by:
      --policy custom --policy_entrypoint examples.custom_policy:make_policy
    """
    return MyPolicy(nvec=np.asarray(nvec, dtype=np.int64), noop=np.asarray(noop, dtype=np.int64), rng=rng)


