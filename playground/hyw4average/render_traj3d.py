#!/usr/bin/env python3
from __future__ import annotations

"""
Render hyw-4average camera trajectory (latents) into a 3D video from a fixed "god view".

Inputs:
  - episode_dir/postprocessed/hyw-4average_camera.json

Outputs:
  - playground/data/<episode_id>_hyw4avg_traj3d.mp4

Notes:
  - Uses Open3D's OffscreenRenderer (real 3D rendering; not matplotlib).
  - Camera JSON indices are *latents* (1 latent per 4 frames).
    Use --latent_hold to slow the video down by repeating each latent frame.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except Exception as e:
        raise SystemExit(
            "Open3D is required for real 3D rendering.\n"
            "Install (in your conda env):\n"
            "  pip install open3d\n"
            f"Original import error: {type(e).__name__}: {e}"
        )


def _load_camera_json(path: Path) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      - latent indices (sorted)
      - w2c list (4x4)
      - K list (3x3)
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("camera json must be a dict")

    keys = sorted((int(k) for k in obj.keys()))
    w2c_list: List[np.ndarray] = []
    K_list: List[np.ndarray] = []
    for k in keys:
        rec = obj[str(k)]
        w2c = np.asarray(rec.get("w2c") or rec.get("extrinsic"), dtype=np.float64)
        K = np.asarray(rec.get("K") or rec.get("intrinsic"), dtype=np.float64)
        if w2c.shape != (4, 4):
            raise ValueError(f"w2c must be 4x4 at {k}, got {w2c.shape}")
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3 at {k}, got {K.shape}")
        w2c_list.append(w2c)
        K_list.append(K)
    return keys, w2c_list, K_list


def _invert_w2c(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(w2c)


def _camera_center_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3].astype(np.float64)


def _make_frustum_lines(K: np.ndarray, *, w: int, h: int, z: float = 1.0) -> np.ndarray:
    """
    Build 5 frustum points in camera coordinates:
      - origin
      - 4 image corners back-projected to depth z
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    corners = [
        (0.0, 0.0),
        (float(w), 0.0),
        (float(w), float(h)),
        (0.0, float(h)),
    ]

    pts = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    for (u, v) in corners:
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pts.append(np.array([x, y, z], dtype=np.float64))
    return np.stack(pts, axis=0)  # (5,3)


def _transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1)  # (N,4)
    out = (T @ ph.T).T[:, :3]
    return out


def _compute_god_view(points: np.ndarray, *, azim_deg: float, elev_deg: float, dist_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (eye, center, up).
    """
    center = points.mean(axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    dist = max(1.0, diag * dist_scale)

    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    eye = center + np.array(
        [
            dist * math.cos(el) * math.cos(az),
            dist * math.sin(el),
            dist * math.cos(el) * math.sin(az),
        ],
        dtype=np.float64,
    )
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return eye, center, up


def _make_checkerboard(
    *,
    size: float,
    squares_per_side: int = 40,
    y: float = 0.0,
    color_a: Tuple[float, float, float] = (0.92, 0.92, 0.92),
    color_b: Tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> "Any":
    """
    Create a checkerboard plane (TriangleMesh) on XZ with vertex colors.
    """
    o3d = _require_open3d()
    n = int(squares_per_side)
    n = max(2, n)
    half = float(size) / 2.0
    step = float(size) / n

    verts: List[List[float]] = []
    tris: List[List[int]] = []
    cols: List[List[float]] = []

    for ix in range(n):
        for iz in range(n):
            x0 = -half + ix * step
            x1 = x0 + step
            z0 = -half + iz * step
            z1 = z0 + step

            base = len(verts)
            verts.extend([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]])
            # Two-sided: add both winding orders so the plane is visible even if back-face culled.
            tris.extend(
                [
                    [base + 0, base + 1, base + 2],
                    [base + 0, base + 2, base + 3],
                    [base + 2, base + 1, base + 0],
                    [base + 3, base + 2, base + 0],
                ]
            )

            c = color_a if ((ix + iz) % 2 == 0) else color_b
            cols.extend([list(c), list(c), list(c), list(c)])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
    mesh.compute_vertex_normals()
    return mesh


def _camera_basis_from_c2w(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (pos, forward, up) in world coordinates.

    We treat the camera's local +Z as forward (common CV convention).
    We then re-orthonormalize using world up to avoid roll for a nicer 'chase' view.
    """
    pos = c2w[:3, 3].astype(np.float64)
    forward = c2w[:3, 2].astype(np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-9)

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), forward)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-9)
    return pos, forward, up


def _rotation_matrix_from_z_to_vec(v: np.ndarray) -> np.ndarray:
    """
    Rotation that maps +Z axis to direction v (world).
    """
    v = v.astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.cross(z, v)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    axis = axis / s
    angle = math.atan2(s, c)
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R


def _make_comet_trail_mesh(
    points: List[np.ndarray],
    *,
    radius_max: float,
    radius_min: float,
    resolution: int = 12,
    color_head: Tuple[float, float, float] = (0.05, 0.55, 0.95),
    color_tail: Tuple[float, float, float] = (0.75, 0.85, 0.95),
) -> "Any":
    """
    Build a tube-like trail from a list of positions by chaining cylinders.
    Oldest->newest radius increases (comet head).
    """
    o3d = _require_open3d()
    if len(points) < 2:
        return o3d.geometry.TriangleMesh()

    n_seg = len(points) - 1
    meshes = []
    for i in range(n_seg):
        p0 = points[i]
        p1 = points[i + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        t = i / max(1, (n_seg - 1))  # 0 oldest -> 1 newest
        r = float(radius_min + (radius_max - radius_min) * t)
        c = tuple(float(color_tail[j] * (1.0 - t) + color_head[j] * t) for j in range(3))

        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L, resolution=int(resolution), split=1)
        cyl.compute_vertex_normals()
        cyl.paint_uniform_color(list(c))

        R = _rotation_matrix_from_z_to_vec(d)
        cyl.rotate(R, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
        cyl.translate(((p0 + p1) * 0.5).tolist())
        meshes.append(cyl)

    if not meshes:
        return o3d.geometry.TriangleMesh()
    out = meshes[0]
    for m in meshes[1:]:
        out += m
    out.compute_vertex_normals()
    return out


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    """
    m = R.astype(np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return q


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Quaternion [w,x,y,z] -> 3x3 rotation matrix.
    """
    w, x, y, z = [float(v) for v in q]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )
    return R


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.
    """
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return (s0 * q0) + (s1 * q1)


def _interp_c2w(c2w0: np.ndarray, c2w1: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolate camera pose in SE(3):
    - translation: linear
    - rotation: quaternion slerp
    """
    R0 = c2w0[:3, :3]
    R1 = c2w1[:3, :3]
    q0 = _rotmat_to_quat(R0)
    q1 = _rotmat_to_quat(R1)
    q = _slerp(q0, q1, float(t))
    R = _quat_to_rotmat(q)
    p0 = c2w0[:3, 3]
    p1 = c2w1[:3, 3]
    p = (1.0 - t) * p0 + t * p1
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = p
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Render hyw-4average camera trajectory into a 3D video (Open3D).")
    p.add_argument("--episode_dir", type=str, required=True, help="Path to episode directory containing postprocessed/")
    p.add_argument(
        "--camera_json",
        type=str,
        default=None,
        help="Optional explicit camera json path (defaults to <episode_dir>/postprocessed/hyw-4average_camera.json).",
    )
    p.add_argument("--out_dir", type=str, default="playground/data", help="Output directory (will be created).")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=25)
    # Back-compat: latent_hold used to repeat frames; we keep it but default to 1.
    p.add_argument("--latent_hold", type=int, default=1, help="(legacy) Repeat each latent for N identical frames.")
    p.add_argument(
        "--frames_per_latent",
        type=int,
        default=4,
        help="How many output frames to generate per latent. Default 4 matches 4-frame latent at fps=25.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="interp",
        choices=["interp", "repeat"],
        help="interp: interpolate pose inside each latent; repeat: duplicate the latent frame.",
    )

    p.add_argument("--azim_deg", type=float, default=45.0, help="God view azimuth (degrees).")
    p.add_argument("--elev_deg", type=float, default=30.0, help="God view elevation (degrees).")
    p.add_argument("--dist_scale", type=float, default=2.5, help="God view distance multiplier relative to path bbox diag.")
    p.add_argument(
        "--view",
        type=str,
        default="follow",
        choices=["follow", "god"],
        help="Render viewpoint: follow (chase camera) or god (fixed overview).",
    )
    p.add_argument("--follow_dist", type=float, default=3.0, help="Chase camera distance behind the camera.")
    p.add_argument("--follow_height", type=float, default=1.2, help="Chase camera height offset.")
    p.add_argument("--look_ahead", type=float, default=1.0, help="How far ahead of the camera to look (when follow_look=ahead).")
    p.add_argument(
        "--follow_look",
        type=str,
        default="camera",
        choices=["camera", "ahead"],
        help="Third-person follow: look at camera object (camera) or look ahead along camera forward (ahead).",
    )
    p.add_argument(
        "--follow_look_down",
        type=float,
        default=0.4,
        help="Tilt the third-person camera downward by shifting look-at point down (in world Y).",
    )
    p.add_argument(
        "--follow_smoothing",
        type=float,
        default=0.12,
        help="EMA smoothing for follow camera (0=no smoothing, closer to 1=more smoothing).",
    )
    p.add_argument(
        "--checker_square_size",
        type=float,
        default=2.0,
        help="Checker square size in world units (smaller => higher density).",
    )
    p.add_argument(
        "--checker_size_scale",
        type=float,
        default=40.0,
        help="Checkerboard size = max(200, diag*scale).",
    )
    p.add_argument(
        "--trail_len",
        type=int,
        default=60,
        help="Number of recent rendered frames to keep in comet trail.",
    )
    p.add_argument("--trail_radius_max", type=float, default=0.20)
    p.add_argument("--trail_radius_min", type=float, default=0.04)
    p.add_argument("--trail_resolution", type=int, default=12)
    p.add_argument(
        "--with_rgb",
        action="store_true",
        help="Compose final video with original episode video on the right (height-matched).",
    )
    p.add_argument(
        "--rgb_path",
        type=str,
        default=None,
        help="Optional explicit path to original RGB video (defaults to <episode_dir>/video.mp4).",
    )

    p.add_argument("--frustum_depth", type=float, default=1.0, help="Depth (in camera coords) for drawing frustum.")
    p.add_argument("--draw_every", type=int, default=1, help="Draw frustum every N latents (1=all).")

    args = p.parse_args()

    o3d = _require_open3d()

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    cam_path = (
        Path(args.camera_json).expanduser().resolve()
        if args.camera_json
        else (episode_dir / "postprocessed" / "hyw-4average_camera.json")
    )
    if not cam_path.exists():
        raise SystemExit(f"camera json not found: {cam_path}")

    keys, w2c_list, K_list = _load_camera_json(cam_path)

    # compute camera centers for trajectory
    c2w_list = [_invert_w2c(w2c) for w2c in w2c_list]
    centers = np.stack([_camera_center_from_c2w(c2w) for c2w in c2w_list], axis=0)

    eye, center, up = _compute_god_view(centers, azim_deg=args.azim_deg, elev_deg=args.elev_deg, dist_scale=args.dist_scale)

    # Prefer OffscreenRenderer, but macOS often doesn't support EGL headless.
    use_offscreen = True
    renderer = None
    scene = None
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(int(args.width), int(args.height))
        scene = renderer.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])
    except Exception as e:
        use_offscreen = False
        print(f"[warn] OffscreenRenderer unavailable; falling back to Visualizer hidden window. ({type(e).__name__}: {e})")

    # Materials (only used for OffscreenRenderer path)
    mat_line = None
    mat_frustum = None
    mat_mesh = None
    if use_offscreen:
        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0

        mat_frustum = o3d.visualization.rendering.MaterialRecord()
        mat_frustum.shader = "unlitLine"
        mat_frustum.line_width = 1.0

        mat_mesh = o3d.visualization.rendering.MaterialRecord()
        mat_mesh.shader = "defaultLit"
        mat_mesh.base_color = [0.1, 0.6, 0.9, 1.0]

    # Ground checkerboard (mesh)
    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    # Minecraft coordinates are in blocks; local trajectories can be small, so use a large floor by default.
    size = max(200.0, diag * float(args.checker_size_scale))
    floor_y = float(bbox_min[1]) - max(0.25, diag * 0.05)
    # Choose squares_per_side based on desired square size, but clamp for performance.
    sq = max(0.5, float(args.checker_square_size))
    squares_per_side = int(max(20.0, min(240.0, size / sq)))
    checker = _make_checkerboard(size=size, squares_per_side=squares_per_side, y=floor_y)
    if use_offscreen:
        # In Filament scene, colored meshes render well with defaultLit.
        scene.add_geometry("checker", checker, mat_mesh)

    # Full trajectory line (grey)
    traj_lines = np.stack([np.arange(len(centers) - 1), np.arange(1, len(centers))], axis=1).astype(np.int32)
    traj = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(traj_lines),
    )
    traj.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.4, 0.4, 0.4]]), (len(traj_lines), 1)))
    if use_offscreen:
        scene.add_geometry("traj", traj, mat_line)

    # Current marker
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.03, diag * 0.01))
    marker.compute_vertex_normals()
    marker.translate(centers[0].tolist())
    if use_offscreen:
        scene.add_geometry("marker", marker, mat_mesh)

    # Frustum template from first K (assume constant)
    # Use a nominal image size from K principal point if no explicit; approximate w/h as 2*cx,2*cy.
    K0 = K_list[0]
    w_nom = int(round(float(K0[0, 2]) * 2.0))
    h_nom = int(round(float(K0[1, 2]) * 2.0))
    fr_cam = _make_frustum_lines(K0, w=w_nom, h=h_nom, z=float(args.frustum_depth))

    # Build a frustum LineSet (will update points each step)
    fr_lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.zeros((5, 3), dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(fr_lines),
    )
    frustum.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.9, 0.2, 0.2]]), (len(fr_lines), 1)))
    if use_offscreen:
        scene.add_geometry("frustum", frustum, mat_frustum)

    # Set initial camera
    if use_offscreen:
        if args.view == "god":
            scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())
        else:
            # follow view from first pose
            pos0, fwd0, up0 = _camera_basis_from_c2w(c2w_list[0])
            eye0 = pos0 - fwd0 * float(args.follow_dist) + up0 * float(args.follow_height)
            look0 = pos0 if args.follow_look == "camera" else (pos0 + fwd0 * float(args.look_ahead))
            look0 = look0 + np.array([0.0, -float(args.follow_look_down), 0.0], dtype=np.float64)
            scene.camera.look_at(look0.tolist(), eye0.tolist(), up0.tolist())

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_traj = out_dir / f"{episode_dir.name}_hyw4avg_traj3d.mp4"
    out_path_combo = out_dir / f"{episode_dir.name}_hyw4avg_traj3d_with_rgb.mp4"

    # Render frames into memory first (episodes are short by default)
    traj_frames: List[np.ndarray] = []
    try:
        frames_per_latent = max(1, int(args.frames_per_latent))
        legacy_hold = max(1, int(args.latent_hold))

        if use_offscreen:
            for i in range(len(c2w_list)):
                c2w0 = c2w_list[i]
                c2w1 = c2w_list[i + 1] if i + 1 < len(c2w_list) else c2w_list[i]

                if args.mode == "repeat":
                    ts = [0.0] * legacy_hold
                else:
                    # Generate frames_per_latent samples in [0,1), last latent repeats at t=0
                    ts = [(j / frames_per_latent) for j in range(frames_per_latent)]

                for t in ts:
                    c2w = _interp_c2w(c2w0, c2w1, t) if args.mode == "interp" else c2w0
                    pos = _camera_center_from_c2w(c2w)

                    scene.remove_geometry("marker")
                    marker_i = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.03, diag * 0.01))
                    marker_i.compute_vertex_normals()
                    marker_i.translate(pos.tolist())
                    scene.add_geometry("marker", marker_i, mat_mesh)

                    if args.draw_every <= 1 or (i % int(args.draw_every) == 0):
                        fr_w = _transform_points(c2w, fr_cam)
                    else:
                        fr_w = np.zeros((5, 3), dtype=np.float64)

                    scene.remove_geometry("frustum")
                    frustum_i = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(fr_w),
                        lines=o3d.utility.Vector2iVector(fr_lines),
                    )
                    frustum_i.colors = o3d.utility.Vector3dVector(
                        np.tile(np.array([[0.9, 0.2, 0.2]]), (len(fr_lines), 1))
                    )
                    scene.add_geometry("frustum", frustum_i, mat_frustum)

                    img = renderer.render_to_image()
                    traj_frames.append(np.asarray(img))
        else:
            # Hidden-window Visualizer rendering (still real OpenGL rendering).
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="traj3d",
                width=int(args.width),
                height=int(args.height),
                visible=False,
            )
            try:
                vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])
                opt = vis.get_render_option()
                # Ensure vertex colors are used for the checkerboard.
                try:
                    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
                except Exception:
                    pass
                opt.light_on = True
                vis.add_geometry(checker)
                vis.add_geometry(traj)

                # marker + frustum (mutable)
                marker_i = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.03, diag * 0.01))
                marker_i.compute_vertex_normals()
                marker_i.paint_uniform_color([0.1, 0.6, 0.9])
                marker_i.translate(centers[0].tolist())
                vis.add_geometry(marker_i)

                frustum_i = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.zeros((5, 3), dtype=np.float64)),
                    lines=o3d.utility.Vector2iVector(fr_lines),
                )
                frustum_i.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[0.9, 0.2, 0.2]]), (len(fr_lines), 1))
                )
                vis.add_geometry(frustum_i)

                # Configure initial view
                vc = vis.get_view_control()
                if args.view == "god":
                    front = (center - eye).astype(np.float64)
                    front = front / (np.linalg.norm(front) + 1e-9)
                    vc.set_lookat(center.tolist())
                    vc.set_front(front.tolist())
                    vc.set_up(up.tolist())
                    vc.set_zoom(0.7)
                else:
                    pos0, fwd0, up0 = _camera_basis_from_c2w(c2w_list[0])
                    eye0 = pos0 - fwd0 * float(args.follow_dist) + up0 * float(args.follow_height)
                    look0 = pos0 if args.follow_look == "camera" else (pos0 + fwd0 * float(args.look_ahead))
                    look0 = look0 + np.array([0.0, -float(args.follow_look_down), 0.0], dtype=np.float64)
                    # Open3D ViewControl front is the direction FROM lookat TO camera.
                    front0 = (eye0 - look0) / (np.linalg.norm(eye0 - look0) + 1e-9)
                    vc.set_lookat(look0.tolist())
                    vc.set_front(front0.tolist())
                    vc.set_up([0.0, 1.0, 0.0])
                    vc.set_zoom(0.55)

                curr_pos = centers[0].copy()
                # Follow-camera smoothing state
                sm = float(_clamp(args.follow_smoothing, 0.0, 0.95))
                eye_s = None
                look_s = None
                # Comet trail state
                trail_history: List[np.ndarray] = []
                trail_mesh = None
                for i in range(len(c2w_list)):
                    c2w0 = c2w_list[i]
                    c2w1 = c2w_list[i + 1] if i + 1 < len(c2w_list) else c2w_list[i]

                    if args.mode == "repeat":
                        ts = [0.0] * legacy_hold
                    else:
                        ts = [(j / frames_per_latent) for j in range(frames_per_latent)]

                    for t in ts:
                        c2w = _interp_c2w(c2w0, c2w1, t) if args.mode == "interp" else c2w0
                        pos = _camera_center_from_c2w(c2w)
                        if args.view == "follow":
                            pos_b, fwd_b, up_b = _camera_basis_from_c2w(c2w)
                            eye_b = pos_b - fwd_b * float(args.follow_dist) + up_b * float(args.follow_height)
                            look_b = pos_b if args.follow_look == "camera" else (pos_b + fwd_b * float(args.look_ahead))
                            look_b = look_b + np.array([0.0, -float(args.follow_look_down), 0.0], dtype=np.float64)
                            if eye_s is None:
                                eye_s = eye_b
                                look_s = look_b
                            else:
                                eye_s = (1.0 - sm) * eye_s + sm * eye_b
                                look_s = (1.0 - sm) * look_s + sm * look_b
                            eye_use = eye_s
                            look_use = look_s
                            front_b = (eye_use - look_use) / (np.linalg.norm(eye_use - look_use) + 1e-9)
                            vc.set_lookat(look_use.tolist())
                            vc.set_front(front_b.tolist())
                            vc.set_up([0.0, 1.0, 0.0])
                        marker_i.translate((pos - curr_pos).tolist(), relative=True)
                        curr_pos = pos
                        vis.update_geometry(marker_i)

                        # Update comet trail (thick->thin). Keep full grey traj line as background.
                        trail_history.append(pos.copy())
                        if len(trail_history) > int(args.trail_len):
                            trail_history = trail_history[-int(args.trail_len) :]
                        if trail_mesh is not None:
                            try:
                                vis.remove_geometry(trail_mesh, reset_bounding_box=False)
                            except Exception:
                                pass
                        trail_mesh = _make_comet_trail_mesh(
                            trail_history,
                            radius_max=float(args.trail_radius_max),
                            radius_min=float(args.trail_radius_min),
                            resolution=int(args.trail_resolution),
                        )
                        try:
                            vis.add_geometry(trail_mesh, reset_bounding_box=False)
                        except Exception:
                            pass

                        if args.draw_every <= 1 or (i % int(args.draw_every) == 0):
                            fr_w = _transform_points(c2w, fr_cam)
                        else:
                            fr_w = np.zeros((5, 3), dtype=np.float64)
                        frustum_i.points = o3d.utility.Vector3dVector(fr_w)
                        vis.update_geometry(frustum_i)

                        vis.poll_events()
                        vis.update_renderer()
                        img_f = np.asarray(vis.capture_screen_float_buffer(do_render=False))
                        frame = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
                        traj_frames.append(frame)
            finally:
                vis.destroy_window()
    finally:
        pass

    # Write trajectory-only video
    with imageio.get_writer(
        str(out_path_traj),
        fps=int(args.fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        for fr in traj_frames:
            writer.append_data(fr)

    # Optional: side-by-side composition with original episode video on the right
    if bool(args.with_rgb):
        rgb_path = (
            Path(args.rgb_path).expanduser().resolve()
            if args.rgb_path
            else (episode_dir / "video.mp4")
        )
        if not rgb_path.exists():
            raise SystemExit(f"--with_rgb requested but RGB video not found: {rgb_path}")

        from PIL import Image

        def resize_to_h(frame: np.ndarray, target_h: int) -> np.ndarray:
            im = Image.fromarray(frame)
            w, h = im.size
            if h == target_h:
                return np.asarray(im)
            new_w = int(round(w * (target_h / float(h))))
            im2 = im.resize((new_w, target_h), resample=Image.BICUBIC)
            return np.asarray(im2)

        H = int(traj_frames[0].shape[0]) if traj_frames else 0
        if H <= 0:
            raise SystemExit("No trajectory frames rendered; cannot compose.")

        n_out = len(traj_frames)
        rgb_reader = imageio.get_reader(str(rgb_path))
        # Best-effort length; may be inf/unknown for some backends.
        try:
            n_rgb = int(rgb_reader.count_frames())
        except Exception:
            n_rgb = -1

        with imageio.get_writer(
            str(out_path_combo),
            fps=int(args.fps),
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        ) as writer:
            last_rgb = None
            try:
                for i in range(n_out):
                    left = traj_frames[i]
                    try:
                        rgb = rgb_reader.get_data(i)
                        last_rgb = rgb
                    except Exception:
                        if last_rgb is None:
                            raise SystemExit(f"RGB video appears unreadable or empty: {rgb_path}")
                        rgb = last_rgb

                    rgb_r = resize_to_h(np.asarray(rgb), H)
                    combo = np.concatenate([left, rgb_r], axis=1)
                    writer.append_data(combo)
            finally:
                try:
                    rgb_reader.close()
                except Exception:
                    pass

        print(f"Wrote: {out_path_combo}")

    print(f"Wrote: {out_path_traj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


