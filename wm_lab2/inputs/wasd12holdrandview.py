from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

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
    # Pitch bounds (degrees), using obs['location_stats']['pitch'].
    # Convention (also used by mg postprocess): pitch increases as you look down.
    pitch_min_deg: float = -45.0
    pitch_max_deg: float = 45.0
    # MineDojo camera discretization interval in degrees (bin step ~= cam_interval_deg).
    # Used for planning so we don't hit pitch bounds mid-hold.
    cam_interval_deg: float = 15.0

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
        return self._sample_view_delta_with_pitch(pitch_f=None)

    def _sample_view_delta_with_pitch(self, pitch_f: Optional[float]) -> Tuple[int, int]:
        """
        Sample view delta (dpitch_bin, dyaw_bin).

        Planning-only constraint:
        - At the start of each hold block, predict whether applying the sampled dpitch for the
          next `hold_frames` steps would push pitch outside [pitch_min_deg, pitch_max_deg].
        - If so, exclude that direction from sampling.

        This avoids "mid-block ceiling hit then freeze" behavior; we do NOT clamp dpitch per-frame.
        """
        s = int(max(1, self.view_bin_step))

        all_choices: List[Tuple[int, int]] = [
            (-s, 0),  # up
            (s, 0),  # down
            (0, -s),  # left
            (0, s),  # right
            (-s, -s),  # up-left
            (-s, s),  # up-right
            (s, -s),  # down-left
            (s, s),  # down-right
        ]

        choices = all_choices
        if pitch_f is not None and not np.isnan(pitch_f):
            lo = float(self.pitch_min_deg)
            hi = float(self.pitch_max_deg)
            if lo > hi:
                lo, hi = hi, lo

            # Predict end-of-block pitch for each candidate (monotonic within block).
            T = max(1, int(self.hold_frames))
            deg_per_bin = float(self.cam_interval_deg)
            planned: List[Tuple[int, int]] = []
            for dp_bin, dy_bin in all_choices:
                dp_deg_per_frame = float(dp_bin) * deg_per_bin
                pitch_end = float(pitch_f) + dp_deg_per_frame * float(T)
                if lo <= pitch_end <= hi:
                    planned.append((dp_bin, dy_bin))

            if planned:
                choices = planned
            else:
                # If nothing fits (e.g., huge cam_interval or very tight bounds), fall back
                # to yaw-only motion so pitch stays unchanged.
                yaw_only = [(0, -s), (0, s)]
                choices = yaw_only

        return choices[int(self.rng.integers(0, len(choices)))]

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # Read pitch (if available) so we can constrain the next block's view sampling.
        try:
            loc = obs.get("location_stats", {}) if isinstance(obs, dict) else {}
            pitch = loc.get("pitch", None) if isinstance(loc, dict) else None
            pitch_f = float(pitch) if pitch is not None else float("nan")
        except Exception:
            pitch_f = float("nan")

        if self._cached_move is None or self._hold_left <= 0:
            self._cached_move = self._sample_move()
            self._cached_view_delta = self._sample_view_delta_with_pitch(pitch_f=pitch_f)
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


