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
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--latent_hold", type=int, default=4, help="Repeat each latent for N frames to slow playback.")

    p.add_argument("--azim_deg", type=float, default=45.0, help="God view azimuth (degrees).")
    p.add_argument("--elev_deg", type=float, default=30.0, help="God view elevation (degrees).")
    p.add_argument("--dist_scale", type=float, default=2.5, help="God view distance multiplier relative to path bbox diag.")

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

    # Ground grid (thin box + line grid)
    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    size = max(5.0, diag * 1.2)
    grid_n = 20

    lines = []
    pts = []
    for i in range(grid_n + 1):
        t = -size / 2 + size * i / grid_n
        pts.append([t, 0.0, -size / 2])
        pts.append([t, 0.0, size / 2])
        lines.append([2 * i, 2 * i + 1])
    base = len(pts)
    for i in range(grid_n + 1):
        t = -size / 2 + size * i / grid_n
        pts.append([-size / 2, 0.0, t])
        pts.append([size / 2, 0.0, t])
        lines.append([base + 2 * i, base + 2 * i + 1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    grid.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.85, 0.85, 0.85]]), (len(lines), 1)))
    if use_offscreen:
        scene.add_geometry("grid", grid, mat_line)

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

    # Set god-view camera
    if use_offscreen:
        scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode_dir.name}_hyw4avg_traj3d.mp4"

    # Render frames
    writer = imageio.get_writer(
        str(out_path),
        fps=int(args.fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    try:
        if use_offscreen:
            for i, c2w in enumerate(c2w_list):
                # Update marker position (recreate; simplest for Filament scene)
                pos = centers[i]
                scene.remove_geometry("marker")
                marker_i = o3d.geometry.TriangleMesh.create_sphere(radius=max(0.03, diag * 0.01))
                marker_i.compute_vertex_normals()
                marker_i.translate(pos.tolist())
                scene.add_geometry("marker", marker_i, mat_mesh)

                # Update frustum points (optionally sparsify)
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
                frame = np.asarray(img)
                for _ in range(max(1, int(args.latent_hold))):
                    writer.append_data(frame)
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
                vis.add_geometry(grid)
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

                # Configure god view
                vc = vis.get_view_control()
                front = (center - eye).astype(np.float64)
                front = front / (np.linalg.norm(front) + 1e-9)
                vc.set_lookat(center.tolist())
                vc.set_front(front.tolist())
                vc.set_up(up.tolist())
                vc.set_zoom(0.7)

                curr_pos = centers[0].copy()
                for i, c2w in enumerate(c2w_list):
                    pos = centers[i]
                    marker_i.translate((pos - curr_pos).tolist(), relative=True)
                    curr_pos = pos
                    vis.update_geometry(marker_i)

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
                    for _ in range(max(1, int(args.latent_hold))):
                        writer.append_data(frame)
            finally:
                vis.destroy_window()
    finally:
        writer.close()

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


