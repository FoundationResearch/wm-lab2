#!/usr/bin/env python3
from __future__ import annotations

"""
Batch render all episodes under data/<folder>/ into 3D trajectory videos.

Example:
  conda activate alexwm2
  python playground/hyw4average/render_all.py --data_dir ./data/wasd4hold --with_rgb
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _find_episode_dirs(data_dir: Path) -> List[Path]:
    return sorted([p for p in data_dir.glob("episode_*") if p.is_dir()])


def _has_hyw_camera(ep_dir: Path) -> bool:
    return (ep_dir / "postprocessed" / "hyw-4average_camera.json").exists()


def _run(cmd: List[str]) -> int:
    return subprocess.call(cmd)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch render hyw-4average trajectory videos for all episodes in a folder.")
    p.add_argument("--data_dir", type=str, required=True, help="Folder like ./data/<policy>/ containing episode_* dirs.")
    p.add_argument("--out_dir", type=str, default="./playground/data", help="Where to write mp4 outputs.")
    p.add_argument("--auto_postprocess", action="store_true", help="If set, run hyw-4average postprocess before rendering.")
    p.add_argument("--with_rgb", action="store_true", help="Compose with original episode video on the right.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (passed through to renderer).")

    # Pass-through knobs (optional; keep defaults in render_traj3d.py)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frames_per_latent", type=int, default=4)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fog", action="store_true")
    p.add_argument("--fog_start", type=float, default=25.0)
    p.add_argument("--fog_end", type=float, default=200.0)
    p.add_argument("--z_far", type=float, default=8000.0)

    args = p.parse_args(list(argv) if argv is not None else None)

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"data_dir not found: {data_dir}", file=sys.stderr)
        return 2

    eps = _find_episode_dirs(data_dir)
    if not eps:
        print(f"No episode_* directories found under: {data_dir}", file=sys.stderr)
        return 1

    # Optionally run postprocess once for the whole folder.
    if args.auto_postprocess:
        cmd = [
            sys.executable,
            "-m",
            "wm_lab2.tools.postprocess",
            "--root",
            str(data_dir),
            "--only_ok",
            "--processor",
            "hyw-4average",
            "--out_mode",
            "subdir",
            "--overwrite",
        ]
        print("[postprocess]", " ".join(cmd))
        rc = _run(cmd)
        if rc != 0:
            print(f"postprocess failed with code {rc}", file=sys.stderr)
            return rc

    renderer = Path(__file__).parent / "render_traj3d.py"
    if not renderer.exists():
        print(f"renderer not found: {renderer}", file=sys.stderr)
        return 2

    ok = 0
    skipped = 0
    failed = 0
    for ep in eps:
        if not _has_hyw_camera(ep):
            print(f"[skip] missing postprocessed/hyw-4average_camera.json: {ep.name}")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(renderer),
            "--episode_dir",
            str(ep),
            "--out_dir",
            str(out_dir),
            "--fps",
            str(int(args.fps)),
            "--mode",
            "interp",
            "--frames_per_latent",
            str(int(args.frames_per_latent)),
            "--view",
            "follow",
            "--follow_look",
            "camera",
            "--z_far",
            str(float(args.z_far)),
            "--width",
            str(int(args.width)),
            "--height",
            str(int(args.height)),
        ]
        if args.fog:
            cmd += ["--fog", "--fog_start", str(float(args.fog_start)), "--fog_end", str(float(args.fog_end))]
        if args.with_rgb:
            cmd += ["--with_rgb"]

        # overwrite isn't currently a renderer flag; outputs are deterministic by episode name and will be overwritten
        print("[render]", ep.name)
        rc = _run(cmd)
        if rc == 0:
            ok += 1
        else:
            failed += 1

    print(f"done: ok={ok}, skipped={skipped}, failed={failed}, total={len(eps)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


