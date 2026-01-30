from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np


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
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


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


