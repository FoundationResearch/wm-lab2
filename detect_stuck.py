#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


def _iter_npz_paths(root: Path) -> Iterable[Path]:
    """
    Yield episode data npz paths under root.
    Expected layout:
      <root>/episode_*/data.npz
    """
    if root.is_file() and root.suffix == ".npz":
        yield root
        return

    if not root.exists():
        return

    if root.is_dir():
        # Fast path: only look for episode_*/data.npz
        for ep_dir in sorted(root.glob("episode_*/")):
            p = ep_dir / "data.npz"
            if p.is_file():
                yield p


def _is_stuck_positions(positions, eps: float) -> bool:
    """
    True if the agent never leaves a ball of radius `eps` (meters) centered at the initial position.
    positions: np.ndarray shape (T, 3)
    """
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64)
    if pos.size == 0:
        return True
    if pos.ndim != 2 or pos.shape[1] < 3:
        # Can't reason reliably; treat as not-stuck.
        return False
    p0 = pos[0, :3]
    dif = pos[:, :3] - p0[None, :]
    max_dev = float(np.max(np.linalg.norm(dif, axis=1)))
    return max_dev <= float(eps)


def _tqdm(it, *, total: int, disable: bool):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(it, total=total, disable=disable)
    except Exception:
        return it


def _check_one_npz(args) -> Optional[str]:
    """
    Worker function: returns episode dir path string if stuck else None.
    """
    import numpy as np

    npz_path_s, eps_s = args
    npz_path = Path(npz_path_s)
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "positions" not in z:
                return None
            positions = z["positions"]
    except Exception:
        return None

    try:
        if _is_stuck_positions(positions, eps=float(eps_s)):
            return str(npz_path.parent)
    except Exception:
        return None
    return None


def find_stuck_episodes(*, root: Path, eps: float, workers: int = 32, no_progress: bool = False) -> Tuple[int, int, List[Path]]:
    """
    Returns: (total_npz, stuck, stuck_episode_dirs)
    """
    npz_paths = list(_iter_npz_paths(root))
    total = len(npz_paths)
    if total == 0:
        return 0, 0, []

    w = max(1, int(workers))
    tasks = [(str(p), float(eps)) for p in npz_paths]

    stuck_dirs: List[Path] = []
    if w <= 1:
        it = _tqdm(tasks, total=total, disable=bool(no_progress))
        for t in it:
            out = _check_one_npz(t)
            if out:
                stuck_dirs.append(Path(out))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=w) as pool:
            it = pool.imap_unordered(_check_one_npz, tasks, chunksize=32)
            it = _tqdm(it, total=total, disable=bool(no_progress))
            for out in it:
                if out:
                    stuck_dirs.append(Path(out))

    # Stable ordering for printing/deletion
    stuck_dirs = sorted(set(stuck_dirs))
    return total, len(stuck_dirs), stuck_dirs


def _confirm_or_exit(*, prompt: str) -> None:
    """
    Interactive confirmation. Exits process if not confirmed.
    """
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        raise SystemExit(1)

    if ans not in ("y", "yes"):
        print("Cancelled.", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    p = argparse.ArgumentParser(description="Detect (and optionally delete) stuck episodes by positions in data.npz.")
    p.add_argument(
        "root",
        type=str,
        help="Episode root folder (contains episode_*/data.npz) or a single .npz file path.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="Stuck radius in meters: consider stuck if max distance from initial position <= eps (default: 0.1).",
    )
    p.add_argument("--workers", type=int, default=32, help="Parallel workers for scanning npz files (default: 32).")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    p.add_argument("--stat", action="store_true", help="Print stats about stuck episodes.")
    p.add_argument(
        "--rm",
        action="store_true",
        help="Delete stuck episode folders. Will ask for interactive confirmation at start.",
    )
    p.add_argument("--list", action="store_true", help="List stuck episode folder paths.")

    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()

    if not args.stat and not args.rm and not args.list:
        print("Nothing to do: pass at least one of --stat/--list/--rm", file=sys.stderr)
        return 2

    if bool(args.rm):
        _confirm_or_exit(
            prompt=(
                f"[detect_stuck] You are about to DELETE stuck episodes under:\n"
                f"  {root}\n"
                f"Proceed? (y/N): "
            )
        )

    total, stuck, stuck_dirs = find_stuck_episodes(
        root=root,
        eps=float(args.eps),
        workers=int(args.workers),
        no_progress=bool(args.no_progress),
    )

    if bool(args.stat):
        pct = (100.0 * stuck / total) if total > 0 else 0.0
        print(f"total_npz={total} stuck={stuck} ({pct:.2f}%) eps={args.eps} workers={args.workers}")

    if bool(args.list):
        for d in stuck_dirs:
            print(str(d))

    if bool(args.rm):
        deleted = 0
        for ep_dir in stuck_dirs:
            try:
                shutil.rmtree(ep_dir)
                deleted += 1
            except Exception as e:
                print(f"[detect_stuck] WARN: failed to delete {ep_dir}: {e!r}", file=sys.stderr)
        print(f"deleted={deleted}/{len(stuck_dirs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


