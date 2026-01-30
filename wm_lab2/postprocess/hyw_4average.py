from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from wm_lab2.postprocess.base import Processor


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _deg2rad(d: float) -> float:
    return float(d) * math.pi / 180.0


def _intrinsic_from_fov(
    *,
    width: int,
    height: int,
    fov_deg: float = 70.0,
) -> List[List[float]]:
    """
    Minecraft default horizontal FOV is commonly 70 deg.
    Use: fx = (W/2)/tan(fov/2), fy similarly (approx; assumes square pixels).
    """
    fx = (width / 2.0) / math.tan(_deg2rad(fov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(_deg2rad(fov_deg) / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]


def _w2c_from_pos_yaw_pitch(
    *,
    pos_xyz: Sequence[float],
    yaw_deg: float,
    pitch_deg: float,
) -> List[List[float]]:
    """
    Build a world-to-camera (w2c) 4x4 matrix.

    Conventions (chosen to match typical Minecraft-like yaw/pitch and the sample):
    - yaw: rotation around world up-axis (Y)
    - pitch: rotation around camera X (right) axis
    - We use negative pitch in Rx to match the sample pattern:
        [[1,0,0],
         [0, cos,  sin],
         [0,-sin,  cos]]
    - Then combine yaw and pitch to get R (world->camera).
    - Translation t = -R * C, where C is camera/world position.
    """
    x, y, z = (float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
    yaw = _deg2rad(float(yaw_deg))
    pitch = _deg2rad(float(pitch_deg))

    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)

    # Rx(-pitch)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, sp],
            [0.0, -sp, cp],
        ],
        dtype=np.float64,
    )
    # Ry(-yaw) (negative to better match the sample sign in (0,2))
    Ry = np.array(
        [
            [cy, 0.0, -sy],
            [0.0, 1.0, 0.0],
            [sy, 0.0, cy],
        ],
        dtype=np.float64,
    )

    R = Rx @ Ry  # world -> camera
    C = np.array([[x], [y], [z]], dtype=np.float64)
    t = -(R @ C)  # 3x1

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3:4] = t
    return w2c.tolist()


def _move_action_from_axes(fb: int, lr: int) -> str:
    """
    fb: 0 none, 1 forward(W), 2 backward(S)
    lr: 0 none, 1 left(A),    2 right(D)
    Returns "", "W","A","S","D","WA","WD","SA","SD"
    """
    fb_map = {0: "", 1: "W", 2: "S"}
    lr_map = {0: "", 1: "A", 2: "D"}
    a = fb_map.get(int(fb), "")
    b = lr_map.get(int(lr), "")
    return f"{a}{b}"


def _view_action_from_yaw_pitch_delta(
    dyaw: float,
    dpitch: float,
    *,
    deadzone_deg: float = 1e-3,
) -> str:
    """
    Quantize yaw/pitch change into a single-direction view action (plus empty).

    Codebook (matches your one-hot mapping):
    - ""  : no significant change
    - "LR": look right
    - "LL": look left
    - "LU": look up
    - "LD": look down

    If both yaw and pitch change, we pick the dominant axis by absolute magnitude.
    """
    if abs(dyaw) < deadzone_deg and abs(dpitch) < deadzone_deg:
        return ""

    # Choose dominant axis if both change.
    use_yaw = abs(dyaw) >= abs(dpitch)

    if use_yaw and abs(dyaw) >= deadzone_deg:
        # dyaw > 0 => right
        return "LR" if dyaw > 0 else "LL"

    if abs(dpitch) >= deadzone_deg:
        # Minecraft pitch typically increases when looking DOWN.
        return "LD" if dpitch > 0 else "LU"

    return ""


@dataclass
class HYW4AverageProcessor(Processor):
    """
    hyw-4average:
    - Group frames by 4: (0..3)->latent 0, (4..7)->latent 1, ...
    - For each group, majority-vote the parsed move_action; view_action is derived from
      the group's yaw/pitch delta (start -> end) and quantized.
    - Write per-latent camera intrinsics/extrinsics using a representative pose
      (default: last frame in the group).
    """

    name: str = "hyw-4average"
    window: int = 4
    deadzone_deg: float = 1e-3
    fov_deg: float = 70.0
    drop_tail: bool = False  # if True, ignore last group if it has < window frames
    pose_frame: str = "last"  # "first" | "last"
    # Output filenames
    actions_filename: str = "hyw-4average_actions.json"
    camera_filename: str = "hyw-4average_camera.json"

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

        if actions.ndim != 2:
            raise ValueError(f"actions must be 2D, got shape {actions.shape}")
        T = int(actions.shape[0])
        if positions.shape[0] != T or yaws.shape[0] != T or pitches.shape[0] != T:
            raise ValueError("positions/yaws/pitches must align with actions on axis 0")

        # Determine image size from manifest if possible.
        img_hw = manifest.get("image_size_hw")
        if isinstance(img_hw, list) and len(img_hw) == 2:
            H, W = int(img_hw[0]), int(img_hw[1])
        else:
            # fallback
            H, W = 160, 256

        K = _intrinsic_from_fov(width=W, height=H, fov_deg=self.fov_deg)

        # ---- per-frame parsed move actions (for voting) ----
        move_per_frame: List[str] = []
        for t in range(T):
            fb = int(actions[t, 0]) if actions.shape[1] > 0 else 0
            lr = int(actions[t, 1]) if actions.shape[1] > 1 else 0
            move_per_frame.append(_move_action_from_axes(fb, lr))

        # ---- group into latents (1 latent per 4 frames) ----
        n_latents = (T // self.window) if self.drop_tail else ((T + self.window - 1) // self.window)

        action_out: Dict[str, Dict[str, str]] = {}
        cam_out: Dict[str, Dict[str, Any]] = {}

        for li in range(n_latents):
            start = li * self.window
            end_excl = min(T, start + self.window)
            if self.drop_tail and (end_excl - start) < self.window:
                break

            # majority vote for move_action within the group
            moves = move_per_frame[start:end_excl]
            move_vote = Counter(moves).most_common(1)[0][0] if moves else ""

            # view_action from yaw/pitch delta within the group (start -> last)
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

            # representative pose for camera
            rep_idx = start if self.pose_frame == "first" else (end_excl - 1)
            w2c = _w2c_from_pos_yaw_pitch(
                pos_xyz=positions[rep_idx].tolist(),
                yaw_deg=float(yaws[rep_idx]),
                pitch_deg=float(pitches[rep_idx]),
            )
            cam_out[str(li)] = {
                "intrinsic": K,
                "K": K,
                "w2c": w2c,
                "extrinsic": w2c,
            }

        with actions_path.open("w", encoding="utf-8") as f:
            json.dump(action_out, f, ensure_ascii=False, indent=2)

        with camera_path.open("w", encoding="utf-8") as f:
            json.dump(cam_out, f, ensure_ascii=False, indent=2)


