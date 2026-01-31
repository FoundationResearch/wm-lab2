from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class WASD12HoldRandViewPolicy(Policy):
    """
    WASD policy with 12-frame hold + random constant-speed view motion per hold block.

    - Movement (WASD + optional jump) is sampled once per block and held for `hold_frames`.
    - View motion: at the start of each block, sample a random direction in the 8-neighborhood
      over (pitch, yaw) and apply a constant delta each frame in that block.

    Implementation detail:
    MineDojo NN action wrapper uses MultiDiscrete camera bins; noop is typically the center bin
    (delta ~0). We apply +/- `view_bin_step` bins around noop each frame.
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator

    hold_frames: int = 12
    p_jump: float = 0.05

    # Indices for MineDojo NNActionSpaceWrapper layout
    idx_forward: int = 0
    idx_strafe: int = 1
    idx_jump: int = 2
    idx_pitch: int = 3
    idx_yaw: int = 4

    # Constant speed in "bins per frame" (1 bin typically == 15 degrees if cam_interval=15)
    view_bin_step: int = 1

    _hold_left: int = 0
    _cached_move: Optional[np.ndarray] = None
    _cached_view_delta: Tuple[int, int] = (0, 0)  # (dpitch_bin, dyaw_bin)

    def reset(self, obs: Dict[str, Any]) -> None:
        self._hold_left = 0
        self._cached_move = None
        self._cached_view_delta = (0, 0)

    def _sample_move(self) -> np.ndarray:
        a = np.array(self.noop, copy=True)

        if 0 <= self.idx_forward < len(a) and int(self.nvec[self.idx_forward]) >= 3:
            a[self.idx_forward] = int(self.rng.choice([0, 1, 2], p=[0.2, 0.6, 0.2]))

        if 0 <= self.idx_strafe < len(a) and int(self.nvec[self.idx_strafe]) >= 3:
            a[self.idx_strafe] = int(self.rng.choice([0, 1, 2], p=[0.7, 0.15, 0.15]))

        if 0 <= self.idx_jump < len(a) and int(self.nvec[self.idx_jump]) > 1:
            if self.rng.random() < self.p_jump:
                a[self.idx_jump] = 1

        return a.astype(np.int64, copy=False)

    def _sample_view_delta(self) -> Tuple[int, int]:
        """
        Returns (dpitch_bin, dyaw_bin) each in {-view_bin_step, 0, +view_bin_step}, not both 0.
        """
        s = int(max(1, self.view_bin_step))
        choices = [
            (-s, 0),  # up
            (s, 0),  # down
            (0, -s),  # left
            (0, s),  # right
            (-s, -s),  # up-left
            (-s, s),  # up-right
            (s, -s),  # down-left
            (s, s),  # down-right
        ]
        return choices[int(self.rng.integers(0, len(choices)))]

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        if self._cached_move is None or self._hold_left <= 0:
            self._cached_move = self._sample_move()
            self._cached_view_delta = self._sample_view_delta()
            self._hold_left = max(1, int(self.hold_frames))

        self._hold_left -= 1

        a = np.array(self._cached_move, copy=True)

        # Apply constant view delta each frame by shifting around noop.
        dp, dy = self._cached_view_delta
        if 0 <= self.idx_pitch < len(a) and int(self.nvec[self.idx_pitch]) > 1:
            base = int(self.noop[self.idx_pitch])
            a[self.idx_pitch] = int(np.clip(base + dp, 0, int(self.nvec[self.idx_pitch]) - 1))
        if 0 <= self.idx_yaw < len(a) and int(self.nvec[self.idx_yaw]) > 1:
            base = int(self.noop[self.idx_yaw])
            a[self.idx_yaw] = int(np.clip(base + dy, 0, int(self.nvec[self.idx_yaw]) - 1))

        return a.astype(np.int64, copy=False)


