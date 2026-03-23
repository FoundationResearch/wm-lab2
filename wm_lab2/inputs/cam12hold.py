from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wm_lab2.inputs.base import Policy


def _sample_view_delta_pitch_yaw(
    *,
    rng: np.random.Generator,
    pitch_f: float,
    hold_frames: int,
    pitch_min_deg: float,
    pitch_max_deg: float,
    cam_interval_deg: float,
    view_bin_step: int,
    mode: str,
) -> Tuple[int, int]:
    """
    Sample (dpitch_bin, dyaw_bin) for one hold block.

    mode:
      - "pitch": only pitch may be non-zero (left/right excluded)
      - "yaw": only yaw may be non-zero
      - "both": 8-neighborhood (same as WASD12HoldRandView)
    """
    s = int(max(1, view_bin_step))

    if mode == "pitch":
        all_choices: List[Tuple[int, int]] = [(-s, 0), (s, 0)]
    elif mode == "yaw":
        all_choices = [(0, -s), (0, s)]
    elif mode == "both":
        all_choices = [
            (-s, 0),
            (s, 0),
            (0, -s),
            (0, s),
            (-s, -s),
            (-s, s),
            (s, -s),
            (s, s),
        ]
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    lo = float(min(pitch_min_deg, pitch_max_deg))
    hi = float(max(pitch_min_deg, pitch_max_deg))
    T = max(1, int(hold_frames))
    deg_per_bin = float(cam_interval_deg)

    if not np.isnan(pitch_f):
        planned: List[Tuple[int, int]] = []
        for dp_bin, dy_bin in all_choices:
            dp_deg_per_frame = float(dp_bin) * deg_per_bin
            pitch_end = float(pitch_f) + dp_deg_per_frame * float(T)
            if lo <= pitch_end <= hi:
                planned.append((dp_bin, dy_bin))
        if planned:
            choices = planned
        else:
            if mode == "yaw":
                choices = all_choices
            elif mode == "both":
                choices = [(0, -s), (0, s)]
            else:
                choices = [(0, 0)]
    else:
        choices = all_choices

    return choices[int(rng.integers(0, len(choices)))]


@dataclass
class Camera12HoldPolicy(Policy):
    """
    Locomotion always no-op; camera moves with constant (dpitch, dyaw) bins per hold block.

    mode: 'pitch' | 'yaw' | 'both' — which camera axes may move (see module docstring).
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator

    mode: str = "both"
    hold_frames: int = 12
    pitch_min_deg: float = -45.0
    pitch_max_deg: float = 45.0
    cam_interval_deg: float = 15.0

    idx_forward: int = 0
    idx_strafe: int = 1
    idx_jump: int = 2
    idx_pitch: int = 3
    idx_yaw: int = 4

    view_bin_step: int = 1

    _hold_left: int = 0
    _cached_view_delta: Tuple[int, int] = (0, 0)

    def reset(self, obs: Dict[str, Any]) -> None:
        self._hold_left = 0
        self._cached_view_delta = (0, 0)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        try:
            loc = obs.get("location_stats", {}) if isinstance(obs, dict) else {}
            pitch = loc.get("pitch", None) if isinstance(loc, dict) else None
            pitch_f = float(pitch) if pitch is not None else float("nan")
        except Exception:
            pitch_f = float("nan")

        if self._hold_left <= 0:
            self._cached_view_delta = _sample_view_delta_pitch_yaw(
                rng=self.rng,
                pitch_f=pitch_f,
                hold_frames=self.hold_frames,
                pitch_min_deg=float(self.pitch_min_deg),
                pitch_max_deg=float(self.pitch_max_deg),
                cam_interval_deg=float(self.cam_interval_deg),
                view_bin_step=int(self.view_bin_step),
                mode=str(self.mode),
            )
            self._hold_left = max(1, int(self.hold_frames))

        self._hold_left -= 1

        a = np.array(self.noop, copy=True)
        dp, dy = self._cached_view_delta

        if 0 <= self.idx_pitch < len(a) and int(self.nvec[self.idx_pitch]) > 1:
            base = int(self.noop[self.idx_pitch])
            a[self.idx_pitch] = int(np.clip(base + dp, 0, int(self.nvec[self.idx_pitch]) - 1))
        if 0 <= self.idx_yaw < len(a) and int(self.nvec[self.idx_yaw]) > 1:
            base = int(self.noop[self.idx_yaw])
            a[self.idx_yaw] = int(np.clip(base + dy, 0, int(self.nvec[self.idx_yaw]) - 1))

        return a.astype(np.int64, copy=False)


@dataclass
class Wasd12HoldXorCamPolicy(Policy):
    """
    Each hold block is either locomotion-only (WASD, optional jump) or camera-only (pitch+yaw),
    never both. Within a block, action is constant for `hold_frames` (default 12).

    Camera-only blocks use the same 8-neighborhood view deltas as WASD12HoldRandView.
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator

    hold_frames: int = 12
    p_jump: float = 0.05
    p_move_block: float = 0.5
    pitch_min_deg: float = -45.0
    pitch_max_deg: float = 45.0
    cam_interval_deg: float = 15.0

    idx_forward: int = 0
    idx_strafe: int = 1
    idx_jump: int = 2
    idx_pitch: int = 3
    idx_yaw: int = 4

    view_bin_step: int = 1

    _hold_left: int = 0
    _use_move: bool = True
    _cached_move: Optional[np.ndarray] = None
    _cached_view_delta: Tuple[int, int] = (0, 0)

    def reset(self, obs: Dict[str, Any]) -> None:
        self._hold_left = 0
        self._use_move = True
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

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        try:
            loc = obs.get("location_stats", {}) if isinstance(obs, dict) else {}
            pitch = loc.get("pitch", None) if isinstance(loc, dict) else None
            pitch_f = float(pitch) if pitch is not None else float("nan")
        except Exception:
            pitch_f = float("nan")

        if self._hold_left <= 0:
            self._use_move = bool(self.rng.random() < float(self.p_move_block))
            if self._use_move:
                self._cached_move = self._sample_move()
                self._cached_view_delta = (0, 0)
            else:
                self._cached_move = np.array(self.noop, copy=True)
                self._cached_view_delta = _sample_view_delta_pitch_yaw(
                    rng=self.rng,
                    pitch_f=pitch_f,
                    hold_frames=self.hold_frames,
                    pitch_min_deg=float(self.pitch_min_deg),
                    pitch_max_deg=float(self.pitch_max_deg),
                    cam_interval_deg=float(self.cam_interval_deg),
                    view_bin_step=int(self.view_bin_step),
                    mode="both",
                )
            self._hold_left = max(1, int(self.hold_frames))

        self._hold_left -= 1

        if self._use_move:
            assert self._cached_move is not None
            return np.array(self._cached_move, copy=True).astype(np.int64, copy=False)

        a = np.array(self.noop, copy=True)
        dp, dy = self._cached_view_delta
        if 0 <= self.idx_pitch < len(a) and int(self.nvec[self.idx_pitch]) > 1:
            base = int(self.noop[self.idx_pitch])
            a[self.idx_pitch] = int(np.clip(base + dp, 0, int(self.nvec[self.idx_pitch]) - 1))
        if 0 <= self.idx_yaw < len(a) and int(self.nvec[self.idx_yaw]) > 1:
            base = int(self.noop[self.idx_yaw])
            a[self.idx_yaw] = int(np.clip(base + dy, 0, int(self.nvec[self.idx_yaw]) - 1))
        return a.astype(np.int64, copy=False)
