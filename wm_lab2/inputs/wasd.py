from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class WASDOnlyPolicy(Policy):
    """
    A simple "WASD (+optional jump)" policy:
    - Only touches locomotion dims (0,1,2 by default)
    - Leaves everything else as no-op to avoid inventory/equip issues

    This is useful when you want a tightly constrained action space.
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator
    p_jump: float = 0.05
    # MineDojo conventions in your `test.py`:
    # - act[0] = 1: forward
    # - act[2] = 1: jump
    idx_forward: int = 0
    idx_strafe: int = 1
    idx_jump: int = 2

    def reset(self, obs: Dict[str, Any]) -> None:
        return None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        a = np.array(self.noop, copy=True)

        # Forward/back/none in {0,1,2} typically where 0=no-op, 1=forward, 2=backward.
        if 0 <= self.idx_forward < len(a) and int(self.nvec[self.idx_forward]) >= 3:
            a[self.idx_forward] = int(self.rng.choice([0, 1, 2], p=[0.2, 0.6, 0.2]))

        # Strafe left/right/none in {0,1,2} typically where 1=left, 2=right (varies by wrapper).
        if 0 <= self.idx_strafe < len(a) and int(self.nvec[self.idx_strafe]) >= 3:
            a[self.idx_strafe] = int(self.rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))

        # Optional jump (only if that dim supports it)
        if 0 <= self.idx_jump < len(a) and int(self.nvec[self.idx_jump]) > 1:
            if self.rng.random() < self.p_jump:
                a[self.idx_jump] = 1

        return a.astype(np.int64, copy=False)


