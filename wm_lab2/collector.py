from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

from wm_lab2.inputs.factory import make_policy
from wm_lab2.output.manifest import build_manifest
from wm_lab2.output.writer import write_episode
from wm_lab2.sim.env import EnvSpec, make_env
from wm_lab2.sim.runner import run_episode
from wm_lab2.utils import ensure_dir, utc_timestamp


@dataclass(frozen=True)
class CollectConfig:
    num_episodes: int = 10
    max_steps: int = 125
    fps: int = 25
    seed: int = 0
    out_root: str = "./dataset"
    no_progress: bool = False
    # Env
    task_id: str = "open-ended"
    image_size_hw: tuple[int, int] = (160, 256)
    cam_interval: float = 15.0
    # Policy
    policy: str = "safe_random"  # safe_random | wasd | custom
    policy_entrypoint: Optional[str] = None  # for custom
    # Policy knobs
    active_dims: Optional[tuple[int, ...]] = None  # for safe_random
    non_stationary_prob: float = 0.95  # for safe_random
    p_jump: float = 0.05  # for wasd
    hold_frames: int = 4  # for wasd4hold


def collect(cfg: CollectConfig) -> int:
    ensure_dir(cfg.out_root)
    rng = np.random.default_rng(cfg.seed)

    env = None
    try:
        env, nvec, noop = make_env(
            EnvSpec(task_id=cfg.task_id, image_size_hw=cfg.image_size_hw, cam_interval=float(cfg.cam_interval))
        )
        policy = make_policy(
            name=cfg.policy,
            nvec=nvec,
            noop=noop,
            rng=rng,
            active_dims=cfg.active_dims,
            non_stationary_prob=cfg.non_stationary_prob,
            p_jump=cfg.p_jump,
            hold_frames=cfg.hold_frames,
            entrypoint=cfg.policy_entrypoint,
        )

        outer = range(cfg.num_episodes)
        if not cfg.no_progress:
            outer = tqdm(outer, desc="Episodes")

        for ep_idx in outer:
            ep_stamp = utc_timestamp()
            episode_dir = os.path.join(cfg.out_root, f"episode_{ep_stamp}_{ep_idx:05d}")
            ensure_dir(episode_dir)

            manifest = build_manifest(
                episode_index=ep_idx,
                timestamp_utc=ep_stamp,
                seed=cfg.seed,
                max_steps=cfg.max_steps,
                fps=cfg.fps,
                task_id=cfg.task_id,
                image_size_hw=cfg.image_size_hw,
                action_nvec=[int(x) for x in nvec.tolist()],
                policy_name=cfg.policy,
            )
            manifest["cam_interval"] = float(cfg.cam_interval)
            # Back-compat keys (kept from original script)
            manifest.setdefault("image_size_hw", [int(cfg.image_size_hw[0]), int(cfg.image_size_hw[1])])
            manifest.setdefault("action_nvec", [int(x) for x in nvec.tolist()])

            with open(os.path.join(episode_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            episode = run_episode(env=env, policy=policy, max_steps=cfg.max_steps)
            write_episode(out_dir=episode_dir, episode=episode, fps=cfg.fps)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return 0


