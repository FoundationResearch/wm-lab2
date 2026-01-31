#!/usr/bin/env python3
"""
MineDojo trajectory data collector (open-ended Minecraft).

This file is intentionally a thin CLI wrapper around the layered implementation:
- input: `wm_lab2.inputs.*` (policies)
- sim:   `wm_lab2.sim.*` (MineDojo env + runner)
- output:`wm_lab2.output.*` (writer + manifest)
"""

from __future__ import annotations

import argparse
import re
from typing import Optional, Tuple

from wm_lab2.collector import CollectConfig, collect


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    # "0,1,2,3,4" -> (0,1,2,3,4)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _sanitize_run_name(name: str) -> str:
    """
    Keep run_name filesystem-friendly:
    - allow: A-Z a-z 0-9 _ - .
    - everything else becomes '-'
    """
    s = str(name).strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
    return s.strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MineDojo trajectories.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel Minecraft env workers.")
    parser.add_argument(
        "--schedule_mode",
        type=str,
        default="dynamic",
        help="Parallel episode scheduling: dynamic (workers pull from shared queue) | static (pre-split evenly).",
    )
    parser.add_argument(
        "--max_episode_retries",
        type=int,
        default=0,
        help="Dynamic schedule only. Retry a failed episode up to N times; 0 means retry forever.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Dynamic schedule only. Treat --num_episodes as the target total and only collect missing episode_index under out_root (matching --run_name when set).",
    )
    parser.add_argument("--max_steps", type=int, default=125)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root. If omitted, defaults to ./data/<policy_name>/",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional tag appended to episode folder names (sanitized to [A-Za-z0-9_.-]).",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")

    # Env
    parser.add_argument("--task_id", type=str, default="open-ended")
    parser.add_argument("--image_size_hw", type=str, default="160,256", help="H,W (MineDojo uses (H,W)).")
    parser.add_argument("--cam_interval", type=float, default=15.0, help="MineDojo camera discretization interval in degrees.")

    # Policy
    parser.add_argument(
        "--policy",
        type=str,
        default="safe_random",
        help="safe_random | wasd | wasd4hold | wasd12hold | wasd12holdrandview | custom",
    )
    parser.add_argument(
        "--policy_entrypoint",
        type=str,
        default=None,
        help="For --policy custom: 'pkg.module:factory' where factory(nvec, noop, rng)->Policy",
    )
    parser.add_argument(
        "--active_dims",
        type=str,
        default=None,
        help="For safe_random: comma-separated dims to randomize (default 0,1,2,3,4).",
    )
    parser.add_argument("--non_stationary_prob", type=float, default=0.95)
    parser.add_argument("--p_jump", type=float, default=0.05)
    parser.add_argument("--hold_frames", type=int, default=4, help="For wasd4hold: hold the sampled WASD action for N frames.")
    parser.add_argument(
        "--pitch_min_deg",
        type=float,
        default=-45.0,
        help="For wasd12holdrandview: minimum allowed camera pitch (deg).",
    )
    parser.add_argument(
        "--pitch_max_deg",
        type=float,
        default=45.0,
        help="For wasd12holdrandview: maximum allowed camera pitch (deg).",
    )
    parser.add_argument(
        "--xvfb_per_worker",
        action="store_true",
        help="Start a dedicated Xvfb server per worker and set DISPLAY accordingly (useful for headless + parallel).",
    )
    parser.add_argument(
        "--xvfb_display_base",
        type=int,
        default=90,
        help="Base X display number for per-worker Xvfb. Worker i uses :<base+i>.",
    )
    parser.add_argument(
        "--minedojo_headless",
        type=str,
        default="auto",
        help="Set MINEDOJO_HEADLESS (auto|0|1). 'auto' sets it when using Xvfb or when DISPLAY is missing.",
    )

    args = parser.parse_args()

    # Policy-specific defaults (only apply if user did not override the defaults).
    policy_l = args.policy.strip().lower()
    if policy_l in ("wasd4hold", "wasd12hold", "wasd12holdrandview"):
        if args.image_size_hw == "160,256":
            args.image_size_hw = "256,256"
        # hold_frames: wasd12hold defaults to 12
        if policy_l in ("wasd12hold", "wasd12holdrandview") and int(args.hold_frames) == 4:
            args.hold_frames = 12
        # view speed: wasd12holdrandview defaults to 5x slower camera (15 -> 3 degrees)
        if policy_l == "wasd12holdrandview" and float(args.cam_interval) == 15.0:
            args.cam_interval = 3.0

    if args.out_root is None:
        args.out_root = f"./data/{policy_l}"

    h, w = _parse_int_tuple(args.image_size_hw)
    active_dims = _parse_int_tuple(args.active_dims) if args.active_dims else None

    cfg = CollectConfig(
        num_episodes=args.num_episodes,
        num_workers=max(1, int(args.num_workers)),
        max_steps=args.max_steps,
        fps=args.fps,
        seed=args.seed,
        out_root=args.out_root,
        run_name=_sanitize_run_name(args.run_name),
        no_progress=args.no_progress,
        schedule_mode=str(args.schedule_mode).strip().lower(),
        max_episode_retries=int(args.max_episode_retries),
        resume=bool(args.resume),
        task_id=args.task_id,
        image_size_hw=(int(h), int(w)),
        cam_interval=float(args.cam_interval),
        policy=args.policy,
        policy_entrypoint=args.policy_entrypoint,
        active_dims=active_dims,
        non_stationary_prob=args.non_stationary_prob,
        p_jump=args.p_jump,
        hold_frames=args.hold_frames,
        pitch_min_deg=float(args.pitch_min_deg),
        pitch_max_deg=float(args.pitch_max_deg),
        xvfb_per_worker=bool(args.xvfb_per_worker),
        xvfb_display_base=int(args.xvfb_display_base),
        minedojo_headless=str(args.minedojo_headless).strip().lower(),
    )
    return collect(cfg)


if __name__ == "__main__":
    raise SystemExit(main())


