from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from wm_lab2.inputs.base import Policy


@dataclass
class StaticPolicy(Policy):
    """
    Do nothing: always return the no-op action.
    """

    nvec: np.ndarray
    noop: np.ndarray
    rng: np.random.Generator

    def reset(self, obs: Dict[str, Any]) -> None:
        return None

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # Always return a copy to avoid accidental in-place mutation downstream.
        return np.array(self.noop, copy=True).astype(np.int64, copy=False)


