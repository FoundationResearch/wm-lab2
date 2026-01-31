from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EnvSpec:
    task_id: str = "open-ended"
    image_size_hw: Tuple[int, int] = (160, 256)
    cam_interval: float = 15.0


def make_env(spec: EnvSpec) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Sim layer: create MineDojo env and return (env, nvec, noop_action).

    We intentionally only rely on:
    - env.action_space.nvec (MultiDiscrete)
    - env.action_space.no_op()
    """
    import minedojo  # type: ignore

    env = minedojo.make(task_id=spec.task_id, image_size=spec.image_size_hw, cam_interval=spec.cam_interval)

    if not hasattr(env.action_space, "nvec"):
        raise RuntimeError(f"Expected env.action_space to have nvec (MultiDiscrete), got: {type(env.action_space)}")
    nvec = np.asarray(env.action_space.nvec, dtype=np.int64)

    if not hasattr(env.action_space, "no_op"):
        raise RuntimeError(
            f"Expected env.action_space to have no_op() for safe action construction, got: {type(env.action_space)}"
        )
    noop = np.asarray(env.action_space.no_op(), dtype=np.int64)

    if noop.shape != nvec.shape:
        # Some wrappers return list; normalize shape.
        noop = noop.reshape(nvec.shape)

    return env, nvec, noop


