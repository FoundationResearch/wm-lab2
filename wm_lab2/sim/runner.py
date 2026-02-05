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
    for _ in range(max(0, int(warmup_steps))):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)  # noqa: F841
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


