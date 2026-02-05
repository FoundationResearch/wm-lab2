from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class FixedMovePolicy(Policy):
    """
    A deterministic locomotion-only policy: always applies fixed forward/strafe/jump values
    and keeps everything else as no-op (including camera).

    MineDojo MultiDiscrete conventions used by this repo:
    - forward axis (idx_forward): 0 none, 1 W, 2 S
    - strafe  axis (idx_strafe): 0 none, 1 A, 2 D
    - jump    axis (idx_jump):   0 none, 1 jump
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator

    forward: int = 0
    strafe: int = 0
    jump: int = 0

    idx_forward: int = 0
    idx_strafe: int = 1
    idx_jump: int = 2

    def reset(self, obs: Dict[str, Any]) -> None:
        return None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        a = np.array(self.noop, copy=True)

        if 0 <= self.idx_forward < len(a) and int(self.nvec[self.idx_forward]) >= 3:
            a[self.idx_forward] = int(np.clip(int(self.forward), 0, int(self.nvec[self.idx_forward]) - 1))

        if 0 <= self.idx_strafe < len(a) and int(self.nvec[self.idx_strafe]) >= 3:
            a[self.idx_strafe] = int(np.clip(int(self.strafe), 0, int(self.nvec[self.idx_strafe]) - 1))

        if 0 <= self.idx_jump < len(a) and int(self.nvec[self.idx_jump]) > 1:
            a[self.idx_jump] = int(np.clip(int(self.jump), 0, int(self.nvec[self.idx_jump]) - 1))

        return a.astype(np.int64, copy=False)


