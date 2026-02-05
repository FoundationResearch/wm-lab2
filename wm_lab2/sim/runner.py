from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from wm_lab2.inputs.base import Policy
from wm_lab2.utils import extract_state, to_hwc_uint8


@dataclass
class EpisodeData:
    frames: List[np.ndarray]  # list of (H, W, 3) uint8
    actions: np.ndarray  # (T, D) int64
    positions: np.ndarray  # (T, 3) float32
    yaws: np.ndarray  # (T,) float32
    pitches: np.ndarray  # (T,) float32


def run_episode(
    env: Any,
    policy: Policy,
    max_steps: int,
    warmup_steps: int = 0,
) -> EpisodeData:
    obs: Dict[str, Any] = env.reset()
    policy.reset(obs)

    frames: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    yaw_list: List[float] = []
    pitch_list: List[float] = []

    # Warmup: step the env without recording (useful to get out of spawn collisions / UI state).
    #
    # Default convention (used by this project):
    # - warmup frame 0: jump once
    # - warmup frames 1..9: hold W (forward)
    # - warmup frames 10..19: wait/noop
    #
    # This is implemented at the runner level so it applies to all policies uniformly and does not
    # depend on policy internal state.
    for wi in range(max(0, int(warmup_steps))):
        # Prefer policy.noop for dimensionality without advancing policy internal state.
        try:
            base = getattr(policy, "noop", None)
            if base is None:
                raise AttributeError
            a = np.array(base, copy=True).astype(np.int64, copy=False)
        except Exception:
            # Fallback: ask the policy for an action (may advance internal state for custom policies).
            a = np.asarray(policy.act(obs), dtype=np.int64).copy()

        if a.ndim != 1:
            a = a.reshape(-1).astype(np.int64, copy=False)

        # Indices follow this repo's convention: forward=0, strafe=1, jump=2.
        if wi == 0:
            if a.shape[0] > 2:
                a[2] = 1
        elif 1 <= wi <= 9:
            if a.shape[0] > 0:
                a[0] = 1
            if a.shape[0] > 1:
                a[1] = 0
            if a.shape[0] > 2:
                a[2] = 0
        else:
            # wait/noop: keep as-is (already noop baseline)
            if a.shape[0] > 2:
                a[2] = 0

        obs, reward, done, info = env.step(a)  # noqa: F841
        if bool(done):
            break

    # Recording phase: collect exactly `max_steps` frames unless the env terminates early.
    for _ in range(max_steps):
        rgb = obs.get("rgb", None)
        if rgb is None:
            raise RuntimeError("Observation missing key 'rgb'.")
        frames.append(to_hwc_uint8(rgb))

        pos, yaw, pitch = extract_state(obs)
        pos_list.append(pos)
        yaw_list.append(yaw)
        pitch_list.append(pitch)

        action = policy.act(obs)
        actions_list.append(np.asarray(action, dtype=np.int64))

        obs, reward, done, info = env.step(action)  # noqa: F841
        if bool(done):
            break

    T = len(frames)
    if T == 0:
        actions = np.zeros((0, 0), dtype=np.int64)
        positions = np.zeros((0, 3), dtype=np.float32)
        yaws = np.zeros((0,), dtype=np.float32)
        pitches = np.zeros((0,), dtype=np.float32)
    else:
        actions = np.stack(actions_list, axis=0).astype(np.int64)
        positions = np.stack(pos_list, axis=0).astype(np.float32)
        yaws = np.asarray(yaw_list, dtype=np.float32)
        pitches = np.asarray(pitch_list, dtype=np.float32)

    if not (T == actions.shape[0] == positions.shape[0] == yaws.shape[0] == pitches.shape[0]):
        raise RuntimeError(
            "Alignment error: "
            f"frames={T}, actions={actions.shape}, positions={positions.shape}, yaws={yaws.shape}, pitches={pitches.shape}"
        )

    return EpisodeData(
        frames=frames,
        actions=actions,
        positions=positions,
        yaws=yaws,
        pitches=pitches,
    )


