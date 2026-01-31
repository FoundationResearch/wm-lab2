from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from wm_lab2.postprocess.base import Processor
from wm_lab2.postprocess.hyw_4average import _intrinsic_from_fov, _view_action_from_yaw_pitch_delta, _w2c_from_pos_yaw_pitch


def _move_action_from_axes(fb: int, lr: int) -> str:
    fb_map = {0: "", 1: "W", 2: "S"}
    lr_map = {0: "", 1: "A", 2: "D"}
    return f"{fb_map.get(int(fb), '')}{lr_map.get(int(lr), '')}"


@dataclass
class MG12AverageProcessor(Processor):
    """
    mg12average:
    - Group frames by 12: (0..11)->latent 0, (12..23)->latent 1, ...
    - move_action: majority vote over the 12 frames (by parsed W/A/S/D/diagonals)
    - view_action: single-direction LR/LL/LU/LD from yaw/pitch delta (start->end) via existing helper
    - camera: per latent K + w2c/extrinsic (representative pose: last frame in group)
    """

    name: str = "mg12average"
    window: int = 12
    fov_deg: float = 70.0
    deadzone_deg: float = 1e-3
    drop_tail: bool = False
    pose_frame: str = "last"  # "first" | "last"

    actions_filename: str = "mg12average_actions.json"
    camera_filename: str = "mg12average_camera.json"

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
        actions_path = out_dir / self.actions_filename
        camera_path = out_dir / self.camera_filename
        if (actions_path.exists() or camera_path.exists()) and not overwrite:
            return

        with np.load(npz_path, allow_pickle=False) as z:
            actions = np.asarray(z["actions"], dtype=np.int64)
            positions = np.asarray(z["positions"], dtype=np.float64)
            yaws = np.asarray(z["yaws"], dtype=np.float64)
            pitches = np.asarray(z["pitches"], dtype=np.float64)

        T = int(actions.shape[0])
        if T == 0:
            return

        img_hw = manifest.get("image_size_hw")
        if isinstance(img_hw, list) and len(img_hw) == 2:
            H, W = int(img_hw[0]), int(img_hw[1])
        else:
            H, W = 160, 256
        K = _intrinsic_from_fov(width=W, height=H, fov_deg=self.fov_deg)

        move_per_frame: List[str] = []
        for t in range(T):
            fb = int(actions[t, 0]) if actions.shape[1] > 0 else 0
            lr = int(actions[t, 1]) if actions.shape[1] > 1 else 0
            move_per_frame.append(_move_action_from_axes(fb, lr))

        n_latents = (T // self.window) if self.drop_tail else ((T + self.window - 1) // self.window)
        action_out: Dict[str, Dict[str, str]] = {}
        cam_out: Dict[str, Dict[str, Any]] = {}

        for li in range(n_latents):
            start = li * self.window
            end_excl = min(T, start + self.window)
            if self.drop_tail and (end_excl - start) < self.window:
                break

            # majority vote for move
            moves = move_per_frame[start:end_excl]
            if moves:
                # simple majority
                move_vote = max(set(moves), key=moves.count)
            else:
                move_vote = ""

            # view from yaw/pitch delta (start -> last)
            if end_excl - start <= 1:
                view = ""
            else:
                y0 = float(yaws[start])
                y1 = float(yaws[end_excl - 1])
                dyaw = y1 - y0
                if dyaw > 180.0:
                    dyaw -= 360.0
                elif dyaw < -180.0:
                    dyaw += 360.0
                dpitch = float(pitches[end_excl - 1] - pitches[start])
                view = _view_action_from_yaw_pitch_delta(dyaw, dpitch, deadzone_deg=self.deadzone_deg)

            action_out[str(li)] = {"move_action": move_vote, "view_action": view}

            rep_idx = start if self.pose_frame == "first" else (end_excl - 1)
            w2c = _w2c_from_pos_yaw_pitch(
                pos_xyz=positions[rep_idx].tolist(),
                yaw_deg=float(yaws[rep_idx]),
                pitch_deg=float(pitches[rep_idx]),
            )
            cam_out[str(li)] = {"intrinsic": K, "K": K, "w2c": w2c, "extrinsic": w2c}

        with actions_path.open("w", encoding="utf-8") as f:
            json.dump(action_out, f, ensure_ascii=False, indent=2)
        with camera_path.open("w", encoding="utf-8") as f:
            json.dump(cam_out, f, ensure_ascii=False, indent=2)


