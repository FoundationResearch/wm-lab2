from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class EpisodeStatus(str, Enum):
    ok = "ok"
    empty = "empty"
    failed = "failed"


@dataclass(frozen=True)
class EpisodePaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def video(self) -> Path:
        return self.root / "video.mp4"

    @property
    def npz(self) -> Path:
        return self.root / "data.npz"


@dataclass
class EpisodeRecord:
    """
    A lightweight view of one episode folder on disk.
    """

    episode_id: str
    paths: EpisodePaths
    manifest: Optional[Dict[str, Any]] = None
    status: EpisodeStatus = EpisodeStatus.failed
    reason: str = ""
    num_steps: Optional[int] = None
    action_dim: Optional[int] = None


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def _inspect_npz(npz_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Returns (T, D, error_reason)
    """
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "actions" not in z:
                return None, None, "npz missing 'actions'"
            actions = z["actions"]
            if actions.ndim != 2:
                return None, None, f"actions ndim={actions.ndim} (expected 2)"
            T, D = int(actions.shape[0]), int(actions.shape[1])

            # Optional alignment checks if present.
            for k, ndim in (("positions", 2), ("yaws", 1), ("pitches", 1)):
                if k in z:
                    arr = z[k]
                    if arr.ndim != ndim:
                        return T, D, f"{k} ndim={arr.ndim} (expected {ndim})"
                    if int(arr.shape[0]) != T:
                        return T, D, f"{k} length {arr.shape[0]} != actions length {T}"

            return T, D, None
    except Exception as e:
        return None, None, f"npz load error: {type(e).__name__}: {e}"


def classify_episode(
    episode_dir: Path,
    *,
    empty_video_bytes: int = 1024,
) -> EpisodeRecord:
    """
    Heuristics:
    - ok: has manifest + video.mp4 (size>threshold) + data.npz readable and T>0
    - empty: missing/too-small video OR npz has T==0 (common for crash at step 0)
    - failed: other inconsistencies (missing files, unreadable npz, etc.)
    """
    ep_id = episode_dir.name
    paths = EpisodePaths(root=episode_dir)
    rec = EpisodeRecord(episode_id=ep_id, paths=paths)

    manifest = _safe_read_json(paths.manifest) if paths.manifest.exists() else None
    rec.manifest = manifest

    video_sz = _file_size(paths.video)
    npz_sz = _file_size(paths.npz)

    if not paths.manifest.exists():
        rec.status = EpisodeStatus.failed
        rec.reason = "missing manifest.json"
        return rec

    if not paths.video.exists() and not paths.npz.exists():
        rec.status = EpisodeStatus.empty
        rec.reason = "missing video.mp4 and data.npz"
        return rec

    if paths.video.exists() and video_sz <= empty_video_bytes:
        rec.status = EpisodeStatus.empty
        rec.reason = f"video too small ({video_sz} bytes)"
        return rec

    if not paths.npz.exists():
        rec.status = EpisodeStatus.failed
        rec.reason = "missing data.npz"
        return rec

    T, D, err = _inspect_npz(paths.npz)
    rec.num_steps = T
    rec.action_dim = D

    if err is not None:
        rec.status = EpisodeStatus.failed
        rec.reason = err
        return rec

    if T is None or T <= 0:
        rec.status = EpisodeStatus.empty
        rec.reason = f"no steps (T={T})"
        return rec

    if not paths.video.exists():
        rec.status = EpisodeStatus.failed
        rec.reason = "missing video.mp4"
        return rec

    rec.status = EpisodeStatus.ok
    rec.reason = "ok"
    return rec


def iter_episode_dirs(root: Path, *, recursive: bool = True) -> Iterable[Path]:
    """
    Yields directories named 'episode_*'.
    """
    if recursive:
        yield from (p for p in root.rglob("episode_*") if p.is_dir())
    else:
        yield from (p for p in root.glob("episode_*") if p.is_dir())


def scan_dataset(
    root: Path,
    *,
    recursive: bool = True,
    empty_video_bytes: int = 1024,
) -> List[EpisodeRecord]:
    recs: List[EpisodeRecord] = []
    for ep_dir in iter_episode_dirs(root, recursive=recursive):
        recs.append(classify_episode(ep_dir, empty_video_bytes=empty_video_bytes))
    recs.sort(key=lambda r: r.episode_id)
    return recs


