from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class WASD4HoldPolicy(Policy):
    """
    WASD policy where the sampled movement is held constant for N frames.

    This is useful when downstream post-processing groups frames into latents
    of size N (e.g. hyw-4average with window=4), so majority voting becomes stable.
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator
    hold_frames: int = 4
    p_jump: float = 0.05
    idx_forward: int = 0
    idx_strafe: int = 1
    idx_jump: int = 2

    _hold_left: int = 0
    _cached_action: Optional[np.ndarray] = None

    def reset(self, obs: Dict[str, Any]) -> None:
        self._hold_left = 0
        self._cached_action = None

    def _sample_wasd_action(self) -> np.ndarray:
        a = np.array(self.noop, copy=True)

        if 0 <= self.idx_forward < len(a) and int(self.nvec[self.idx_forward]) >= 3:
            a[self.idx_forward] = int(self.rng.choice([0, 1, 2], p=[0.2, 0.6, 0.2]))

        if 0 <= self.idx_strafe < len(a) and int(self.nvec[self.idx_strafe]) >= 3:
            a[self.idx_strafe] = int(self.rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))

        if 0 <= self.idx_jump < len(a) and int(self.nvec[self.idx_jump]) > 1:
            if self.rng.random() < self.p_jump:
                a[self.idx_jump] = 1

        return a.astype(np.int64, copy=False)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        if self._cached_action is None or self._hold_left <= 0:
            self._cached_action = self._sample_wasd_action()
            self._hold_left = max(1, int(self.hold_frames))

        self._hold_left -= 1
        return self._cached_action


