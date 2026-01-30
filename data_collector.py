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
from typing import Optional, Tuple

from wm_lab2.collector import CollectConfig, collect


def _parse_int_tuple(s: str) -> Tuple[int, ...]:
    # "0,1,2,3,4" -> (0,1,2,3,4)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MineDojo trajectories.")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=125)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="Output root. If omitted, defaults to ./data/<policy_name>/",
    )
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")

    # Env
    parser.add_argument("--task_id", type=str, default="open-ended")
    parser.add_argument("--image_size_hw", type=str, default="160,256", help="H,W (MineDojo uses (H,W)).")

    # Policy
    parser.add_argument(
        "--policy",
        type=str,
        default="safe_random",
        help="safe_random | wasd | wasd4hold | custom",
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

    args = parser.parse_args()

    # Policy-specific defaults (only apply if user did not override the defaults).
    # - For wasd4hold, we default to 256x256 and output under ./data/hyw4hold
    if args.policy.strip().lower() == "wasd4hold":
        if args.image_size_hw == "160,256":
            args.image_size_hw = "256,256"
        # out_root is always per-policy unless user explicitly overrides it

    if args.out_root is None:
        policy_name = args.policy.strip().lower()
        args.out_root = f"./data/{policy_name}"

    h, w = _parse_int_tuple(args.image_size_hw)
    active_dims = _parse_int_tuple(args.active_dims) if args.active_dims else None

    cfg = CollectConfig(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        seed=args.seed,
        out_root=args.out_root,
        no_progress=args.no_progress,
        task_id=args.task_id,
        image_size_hw=(int(h), int(w)),
        policy=args.policy,
        policy_entrypoint=args.policy_entrypoint,
        active_dims=active_dims,
        non_stationary_prob=args.non_stationary_prob,
        p_jump=args.p_jump,
        hold_frames=args.hold_frames,
    )
    return collect(cfg)


if __name__ == "__main__":
    raise SystemExit(main())


