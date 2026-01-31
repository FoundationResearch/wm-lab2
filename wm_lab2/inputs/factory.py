from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from wm_lab2.inputs.base import Policy
from wm_lab2.inputs.safe_random import SafeRandomPolicy
from wm_lab2.inputs.wasd import WASDOnlyPolicy
from wm_lab2.inputs.wasd4hold import WASD4HoldPolicy


def _import_obj(spec: str) -> Any:
    """
    Import helper: "pkg.module:obj" -> imported object.
    """
    if ":" not in spec:
        raise ValueError(f"Invalid entrypoint '{spec}'. Expected format 'pkg.module:object_name'.")
    mod_name, obj_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    try:
        return getattr(mod, obj_name)
    except AttributeError as e:
        raise ValueError(f"Object '{obj_name}' not found in module '{mod_name}'.") from e


def make_policy(
    *,
    name: str,
    nvec: np.ndarray,
    noop: np.ndarray,
    rng: np.random.Generator,
    # SafeRandom args
    active_dims: Optional[Sequence[int]] = None,
    non_stationary_prob: float = 0.95,
    # WASD args
    p_jump: float = 0.05,
    hold_frames: int = 4,
    # Custom policy factory: "pkg.module:factory"
    entrypoint: Optional[str] = None,
) -> Policy:
    """
    Create a Policy instance.

    Built-ins:
    - name="safe_random"
    - name="wasd"
    - name="wasd4hold"
    - name="wasd12hold"

    Custom:
    - name="custom" + entrypoint="pkg.module:factory"
      where factory signature is: factory(nvec, noop, rng) -> Policy
    """
    name = name.strip().lower()

    if name == "safe_random":
        dims = tuple(active_dims) if active_dims is not None else (0, 1, 2, 3, 4)
        return SafeRandomPolicy(
            nvec=nvec,
            noop=noop,
            rng=rng,
            active_dims=dims,
            non_stationary_prob=non_stationary_prob,
        )

    if name == "wasd":
        return WASDOnlyPolicy(nvec=nvec, noop=noop, rng=rng, p_jump=p_jump)

    if name == "wasd4hold":
        return WASD4HoldPolicy(nvec=nvec, noop=noop, rng=rng, p_jump=p_jump, hold_frames=hold_frames)

    if name == "wasd12hold":
        return WASD4HoldPolicy(nvec=nvec, noop=noop, rng=rng, p_jump=p_jump, hold_frames=hold_frames)

    if name == "custom":
        if not entrypoint:
            raise ValueError("policy 'custom' requires --policy_entrypoint pkg.module:factory")
        obj = _import_obj(entrypoint)
        if not callable(obj):
            raise ValueError(f"Custom policy entrypoint '{entrypoint}' is not callable.")
        policy = obj(nvec, noop, rng)
        if not hasattr(policy, "act"):
            raise ValueError("Custom factory must return a Policy-like object with .act(obs)->action.")
        return policy  # type: ignore[return-value]

    raise ValueError(f"Unknown policy '{name}'. Use one of: safe_random, wasd, wasd4hold, wasd12hold, custom.")


