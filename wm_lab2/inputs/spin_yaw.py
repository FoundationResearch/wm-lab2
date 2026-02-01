from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class YawSpinPolicy(Policy):
    """
    Deterministic debug policy:
    - No movement (noop for WASD/jump)
    - Constant yaw turn: +`yaw_bin_step` bins every frame

    Notes:
    - MineDojo's NN action wrapper uses MultiDiscrete camera bins.
      The "noop" action is typically the center bin (delta ~0).
      We set yaw = noop_yaw + yaw_bin_step each frame.
    - If cam_interval=1 (degrees), yaw_bin_step=1 => ~1 degree per frame.
    """

    nvec: np.ndarray
    noop: np.ndarray
    yaw_bin_step: int = 1

    # MineDojo NNActionSpaceWrapper layout (matches other policies in this repo)
    idx_pitch: int = 3
    idx_yaw: int = 4

    def reset(self, obs: Dict[str, Any]) -> None:
        # Stateless
        return None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        a = np.array(self.noop, copy=True)

        # Keep pitch at noop; apply constant yaw delta each frame.
        if 0 <= self.idx_yaw < len(a) and int(self.nvec[self.idx_yaw]) > 1:
            base = int(self.noop[self.idx_yaw])
            step = int(self.yaw_bin_step)
            a[self.idx_yaw] = int(np.clip(base + step, 0, int(self.nvec[self.idx_yaw]) - 1))

        if 0 <= self.idx_pitch < len(a) and int(self.nvec[self.idx_pitch]) > 1:
            a[self.idx_pitch] = int(self.noop[self.idx_pitch])

        return a.astype(np.int64, copy=False)


def factory(nvec: np.ndarray, noop: np.ndarray, rng: np.random.Generator) -> Policy:
    # rng unused (deterministic)
    _ = rng
    return YawSpinPolicy(nvec=np.asarray(nvec, dtype=np.int64), noop=np.asarray(noop, dtype=np.int64), yaw_bin_step=1)


