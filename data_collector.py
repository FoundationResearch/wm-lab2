#!/usr/bin/env python3
"""
MineDojo trajectory data collector (open-ended Minecraft).

Saves per-episode:
- video.mp4: H.264-compressed RGB frames (H, W, 3), uint8
- data.npz : aligned step-wise arrays: actions, positions, yaws, pitches

Environment:
  minedojo.make(task_id="open-ended", image_size=(160, 256))
  NOTE: image_size is (H, W) in MineDojo.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
from tqdm import tqdm


def utc_timestamp() -> str:
    # Example: 20260129T153012Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_hwc_uint8(rgb: np.ndarray) -> np.ndarray:
    """
    Convert MineDojo obs['rgb'] to (H, W, 3) uint8 for video writing.

    Common cases:
    - Already HWC: (H, W, 3)
    - Channel-first: (3, H, W) -> transpose to HWC
    """
    if not isinstance(rgb, np.ndarray):
        rgb = np.asarray(rgb)

    if rgb.ndim != 3:
        raise ValueError(f"Expected rgb with 3 dims, got shape {rgb.shape}")

    # If channel-first (3, H, W), convert to channel-last (H, W, 3).
    if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.shape[-1] != 3:
        raise ValueError(f"Expected rgb last dim == 3, got shape {rgb.shape}")

    if rgb.dtype != np.uint8:
        # MineDojo typically provides uint8; if not, clip and convert safely.
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


@dataclass
class RandomPolicy:
    """
    Random policy for MineDojo MultiDiscrete actions with a mild bias toward
    movement + camera changes to avoid standing still.

    MineDojo action_space is typically gym.spaces.MultiDiscrete with:
      - action_space.nvec: per-dimension cardinalities
      - a sampled action is a vector of ints, shape (D,)
    """

    nvec: np.ndarray
    rng: np.random.Generator
    # Probability of forcing some non-zero movement/look, to avoid "do nothing".
    non_stationary_prob: float = 0.95

    def sample(self) -> np.ndarray:
        a = np.array([self.rng.integers(0, int(n)) for n in self.nvec], dtype=np.int64)

        # Heuristic: with high probability, enforce at least one dimension != 0.
        # This is robust across unknown action layouts: "0" is commonly NOOP.
        if self.rng.random() < self.non_stationary_prob and np.all(a == 0):
            idx = int(self.rng.integers(0, len(a)))
            if self.nvec[idx] > 1:
                a[idx] = int(self.rng.integers(1, int(self.nvec[idx])))
        return a


def extract_state(obs: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
    """
    Extract:
      - position: (3,) float32
      - yaw: float32
      - pitch: float32

    Expected MineDojo keys (typical):
      obs['location_stats']['pos'] -> (3,)
      obs['location_stats']['yaw'] -> scalar
      obs['location_stats']['pitch'] -> scalar
    Falls back to NaNs if missing.
    """
    loc = obs.get("location_stats", {}) if isinstance(obs, dict) else {}

    pos = loc.get("pos", None)
    if pos is None:
        pos_arr = np.full((3,), np.nan, dtype=np.float32)
    else:
        pos_arr = np.asarray(pos, dtype=np.float32).reshape(-1)
        if pos_arr.size >= 3:
            pos_arr = pos_arr[:3]
        else:
            # Pad if needed
            pos_arr = np.pad(pos_arr, (0, 3 - pos_arr.size), constant_values=np.nan).astype(np.float32)

    yaw = loc.get("yaw", np.nan)
    pitch = loc.get("pitch", np.nan)
    try:
        yaw_f = float(yaw)
    except Exception:
        yaw_f = float("nan")
    try:
        pitch_f = float(pitch)
    except Exception:
        pitch_f = float("nan")

    return pos_arr.astype(np.float32), np.float32(yaw_f).item(), np.float32(pitch_f).item()


def write_episode(
    out_dir: str,
    frames: List[np.ndarray],
    actions: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    pitches: np.ndarray,
    fps: int,
) -> None:
    """
    frames: list of (H, W, 3) uint8, length T
    actions: (T, D) int64
    positions: (T, 3) float32
    yaws: (T,) float32
    pitches: (T,) float32
    """
    ensure_dir(out_dir)

    video_path = os.path.join(out_dir, "video.mp4")
    npz_path = os.path.join(out_dir, "data.npz")

    # H.264 video (requires ffmpeg backend available to imageio).
    # - codec libx264: widely supported, good compression
    # - pixelformat yuv420p: maximizes player compatibility
    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,  # avoid forced resizing to multiples of 16
    ) as writer:
        for f in frames:
            writer.append_data(f)

    # Save aligned metadata (compressed).
    np.savez_compressed(
        npz_path,
        actions=actions,
        positions=positions,
        yaws=yaws,
        pitches=pitches,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MineDojo open-ended trajectories.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_root", type=str, default="./dataset")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    # Import here so the script can still be inspected without MineDojo installed.
    import minedojo  # type: ignore

    ensure_dir(args.out_root)

    rng = np.random.default_rng(args.seed)
    env = None

    try:
        env = minedojo.make(task_id="open-ended", image_size=(160, 256))

        # Action space must be MultiDiscrete; we only rely on .nvec to sample correctly.
        if not hasattr(env.action_space, "nvec"):
            raise RuntimeError(
                f"Expected env.action_space to have nvec (MultiDiscrete), got: {type(env.action_space)}"
            )
        nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        policy = RandomPolicy(nvec=nvec, rng=rng)

        outer = range(args.num_episodes)
        if not args.no_progress:
            outer = tqdm(outer, desc="Episodes")

        for ep_idx in outer:
            ep_stamp = utc_timestamp()
            episode_dir = os.path.join(args.out_root, f"episode_{ep_stamp}_{ep_idx:05d}")
            ensure_dir(episode_dir)

            # Store a small manifest for reproducibility/debugging.
            manifest = {
                "episode_index": ep_idx,
                "timestamp_utc": ep_stamp,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "fps": args.fps,
                "image_size_hw": [160, 256],
                "action_nvec": nvec.tolist(),
            }
            with open(os.path.join(episode_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            obs = env.reset()

            frames: List[np.ndarray] = []
            actions_list: List[np.ndarray] = []
            pos_list: List[np.ndarray] = []
            yaw_list: List[float] = []
            pitch_list: List[float] = []

            step_iter = range(args.max_steps)
            if not args.no_progress:
                step_iter = tqdm(step_iter, desc=f"Steps (ep {ep_idx})", leave=False)

            for _ in step_iter:
                # Observation image:
                #   obs['rgb'] is typically uint8 with shape either (H, W, 3) or (3, H, W).
                rgb = obs.get("rgb", None)
                if rgb is None:
                    raise RuntimeError("Observation missing key 'rgb'.")
                frame = to_hwc_uint8(rgb)
                frames.append(frame)

                # State vector:
                pos, yaw, pitch = extract_state(obs)
                pos_list.append(pos)
                yaw_list.append(yaw)
                pitch_list.append(pitch)

                # Action:
                action = policy.sample()  # (D,) int64
                actions_list.append(action)

                obs, reward, done, info = env.step(action)  # noqa: F841
                if bool(done):
                    break

            T = len(frames)
            actions_arr = np.stack(actions_list, axis=0) if T > 0 else np.zeros((0, len(nvec)), dtype=np.int64)
            positions_arr = np.stack(pos_list, axis=0).astype(np.float32) if T > 0 else np.zeros((0, 3), np.float32)
            yaws_arr = np.asarray(yaw_list, dtype=np.float32)
            pitches_arr = np.asarray(pitch_list, dtype=np.float32)

            # Sanity: ensure alignment with video frames.
            if not (len(frames) == actions_arr.shape[0] == positions_arr.shape[0] == yaws_arr.shape[0] == pitches_arr.shape[0]):
                raise RuntimeError(
                    "Alignment error: "
                    f"frames={len(frames)}, actions={actions_arr.shape}, positions={positions_arr.shape}, "
                    f"yaws={yaws_arr.shape}, pitches={pitches_arr.shape}"
                )

            write_episode(
                out_dir=episode_dir,
                frames=frames,
                actions=actions_arr,
                positions=positions_arr,
                yaws=yaws_arr,
                pitches=pitches_arr,
                fps=args.fps,
            )

    finally:
        # Ensure environment is always closed even if collection crashes.
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


