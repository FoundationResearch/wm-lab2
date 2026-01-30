from __future__ import annotations

import os
from typing import Any

import imageio
import numpy as np

from wm_lab2.sim.runner import EpisodeData
from wm_lab2.utils import ensure_dir


def write_episode(
    out_dir: str,
    episode: EpisodeData,
    fps: int,
) -> None:
    """
    Saves:
    - video.mp4: H.264 (frames are uint8 HWC)
    - data.npz : aligned arrays (actions, positions, yaws, pitches)
    """
    ensure_dir(out_dir)

    video_path = os.path.join(out_dir, "video.mp4")
    npz_path = os.path.join(out_dir, "data.npz")

    with imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,  # avoid forced resizing to multiples of 16
    ) as writer:
        for f in episode.frames:
            writer.append_data(f)

    np.savez_compressed(
        npz_path,
        actions=episode.actions,
        positions=episode.positions,
        yaws=episode.yaws,
        pitches=episode.pitches,
    )


