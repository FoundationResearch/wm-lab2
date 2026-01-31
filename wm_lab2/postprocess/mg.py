from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from wm_lab2.postprocess.base import Processor


def _move_action_from_axes(fb: int, lr: int) -> str:
    """
    fb: 0 none, 1 forward(W), 2 backward(S)
    lr: 0 none, 1 left(A),    2 right(D)
    """
    fb_map = {0: "", 1: "W", 2: "S"}
    lr_map = {0: "", 1: "A", 2: "D"}
    return f"{fb_map.get(int(fb), '')}{lr_map.get(int(lr), '')}"


def _view_action_8dir_from_delta(
    dyaw: float,
    dpitch: float,
    *,
    deadzone_deg: float = 1e-3,
) -> str:
    """
    8-direction view action names (plus empty):
      - "" (no significant change)
      - up / down / left / right
      - up_left / up_right / down_left / down_right

    Conventions:
      - dyaw > 0 => right, dyaw < 0 => left
      - dpitch < 0 => up,  dpitch > 0 => down  (Minecraft pitch commonly increases when looking down)
    """
    yaw_ok = abs(dyaw) >= deadzone_deg
    pit_ok = abs(dpitch) >= deadzone_deg
    if not yaw_ok and not pit_ok:
        return ""

    yaw = "right" if dyaw > 0 else "left"
    pit = "down" if dpitch > 0 else "up"

    if yaw_ok and pit_ok:
        return f"{pit}_{yaw}"
    if yaw_ok:
        return yaw
    return pit


@dataclass
class MGPerFrameProcessor(Processor):
    """
    mg:
    - Writes per-frame actions (no temporal aggregation)
    - Adds view_action as 8-direction names (up/down/left/right + diagonals)
    """

    name: str = "mg"
    deadzone_deg: float = 1e-3
    output_filename: str = "mg_actions.json"

    def process(
        self,
        *,
        episode_dir: Path,
        manifest: Dict[str, Any],
        npz_path: Path,
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.output_filename
        if out_path.exists() and not overwrite:
            return

        with np.load(npz_path, allow_pickle=False) as z:
            actions = np.asarray(z["actions"], dtype=np.int64)
            yaws = np.asarray(z["yaws"], dtype=np.float64) if "yaws" in z else None
            pitches = np.asarray(z["pitches"], dtype=np.float64) if "pitches" in z else None

        if actions.ndim != 2:
            raise ValueError(f"actions must be 2D, got shape {actions.shape}")
        T, D = int(actions.shape[0]), int(actions.shape[1])

        out: Dict[str, Dict[str, Any]] = {}
        for t in range(T):
            fb = int(actions[t, 0]) if D > 0 else 0
            lr = int(actions[t, 1]) if D > 1 else 0
            move_action = _move_action_from_axes(fb, lr)

            # view action from yaw/pitch delta if available
            if yaws is None or pitches is None or t == 0:
                view_action = ""
            else:
                dyaw = float(yaws[t] - yaws[t - 1])
                if dyaw > 180.0:
                    dyaw -= 360.0
                elif dyaw < -180.0:
                    dyaw += 360.0
                dpitch = float(pitches[t] - pitches[t - 1])
                view_action = _view_action_8dir_from_delta(dyaw, dpitch, deadzone_deg=self.deadzone_deg)

            out[str(t)] = {
                "action": actions[t].tolist(),
                "move_action": move_action,
                "view_action": view_action,
            }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


