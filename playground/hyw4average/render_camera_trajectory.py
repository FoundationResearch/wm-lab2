from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_camera_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def w2c_to_camera_center(w2c: np.ndarray) -> np.ndarray:
    """
    Given world-to-camera 4x4, recover camera center C in world coords:
      X_c = R * X_w + t
      => C = -R^T * t
    """
    if w2c.shape != (4, 4):
        raise ValueError(f"w2c must be 4x4, got {w2c.shape}")
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    C = -R.T @ t
    return C


def extract_centers(camera_obj: Dict[str, Any]) -> np.ndarray:
    """
    Returns array of shape (N, 3) in world coordinates.
    Keys are latent indices as strings.
    """
    keys = sorted(camera_obj.keys(), key=lambda k: int(k))
    centers: List[np.ndarray] = []
    for k in keys:
        rec = camera_obj[k]
        if not isinstance(rec, dict) or "w2c" not in rec:
            raise ValueError(f"Bad camera record at key {k}")
        w2c = np.asarray(rec["w2c"], dtype=np.float64)
        centers.append(w2c_to_camera_center(w2c))
    return np.stack(centers, axis=0).astype(np.float64)


def _set_axes_equal_3d(ax, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, padding: float) -> None:
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    zmin, zmax = float(zs.min()), float(zs.max())

    xr = xmax - xmin
    yr = ymax - ymin
    zr = zmax - zmin
    r = max(xr, yr, zr) / 2.0 + float(padding)

    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    zmid = (zmin + zmax) / 2.0

    ax.set_xlim(xmid - r, xmid + r)
    ax.set_ylim(ymid - r, ymid + r)
    ax.set_zlim(zmid - r, zmid + r)

    # Matplotlib >=3.3
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def render_trajectory_video_3d(
    centers_xyz: np.ndarray,
    out_mp4: Path,
    *,
    width: int = 768,
    height: int = 768,
    fps: int = 25,
    trail: int = 999999,
    padding: float = 1.0,
    repeat: int = 4,
    elev: float = 45.0,
    azim: float = -60.0,
) -> None:
    """
    Renders a 3D "god view" trajectory video using matplotlib.

    repeat:
      Each latent point will be repeated this many frames in the output video.
      For hyw-4average (4 frames -> 1 latent), set repeat=4 to match original tempo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

    import imageio

    xs = centers_xyz[:, 0]
    ys = centers_xyz[:, 1]
    zs = centers_xyz[:, 2]

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Camera trajectory (3D god view)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_axes_equal_3d(ax, xs, ys, zs, padding=float(padding))
    ax.view_init(elev=float(elev), azim=float(azim))

    (line,) = ax.plot([], [], [], linewidth=2.0, color="tab:blue")
    (pt,) = ax.plot([], [], [], marker="o", markersize=6, color="tab:red")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(out_mp4),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as writer:
        N = int(xs.shape[0])
        rep = max(1, int(repeat))
        for i in range(N):
            j0 = max(0, i - int(trail) + 1)
            line.set_data(xs[j0 : i + 1], ys[j0 : i + 1])
            line.set_3d_properties(zs[j0 : i + 1])
            pt.set_data([xs[i]], [ys[i]])
            pt.set_3d_properties([zs[i]])

            # write multiple frames for this latent step
            for _ in range(rep):
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                # use RGBA buffer (non-deprecated)
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                writer.append_data(buf[:, :, :3])

    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Render hyw-4average camera trajectory from 3D god view.")
    p.add_argument(
        "--episode_dir",
        type=str,
        required=True,
        help="Path to an episode directory containing postprocessed/hyw-4average_camera.json",
    )
    p.add_argument(
        "--camera_json",
        type=str,
        default=None,
        help="Override path to hyw-4average_camera.json (defaults to <episode_dir>/postprocessed/hyw-4average_camera.json)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output mp4 path (default: playground/data/<episode_id>_hyw4avg_traj3d.mp4)",
    )
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--trail", type=int, default=999999, help="How many past points to keep visible.")
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--repeat", type=int, default=4, help="Repeat each latent step this many frames (default 4).")
    p.add_argument("--elev", type=float, default=45.0, help="3D view elevation angle.")
    p.add_argument("--azim", type=float, default=-60.0, help="3D view azimuth angle.")
    args = p.parse_args(list(argv) if argv is not None else None)

    ep = Path(args.episode_dir).expanduser().resolve()
    cam_path = (
        Path(args.camera_json).expanduser().resolve()
        if args.camera_json
        else (ep / "postprocessed" / "hyw-4average_camera.json")
    )
    if args.out:
        out_mp4 = Path(args.out).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[2]
        out_mp4 = (repo_root / "playground" / "data" / f"{ep.name}_hyw4avg_traj3d.mp4").resolve()

    cam_obj = load_camera_json(cam_path)
    centers = extract_centers(cam_obj)
    render_trajectory_video_3d(
        centers,
        out_mp4,
        fps=int(args.fps),
        trail=int(args.trail),
        padding=float(args.padding),
        repeat=int(args.repeat),
        elev=float(args.elev),
        azim=float(args.azim),
    )
    print(f"wrote: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


