from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def build_manifest(
    *,
    episode_index: int,
    timestamp_utc: str,
    seed: int,
    max_steps: int,
    fps: int,
    image_size_hw: Tuple[int, int],
    task_id: str,
    action_nvec: Sequence[int],
    policy_name: str,
) -> Dict[str, Any]:
    return {
        "episode_index": episode_index,
        "timestamp_utc": timestamp_utc,
        "seed": seed,
        "max_steps": max_steps,
        "fps": fps,
        "task_id": task_id,
        "image_size_hw": [int(image_size_hw[0]), int(image_size_hw[1])],
        "action_nvec": [int(x) for x in action_nvec],
        "policy": policy_name,
    }


