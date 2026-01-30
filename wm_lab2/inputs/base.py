from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np


class Policy(Protocol):
    """
    Input layer interface: produce an action vector given the current observation.

    Notes:
    - Works with MineDojo's common MultiDiscrete action representation (np.ndarray shape (D,)).
    - Policies may keep internal state; call reset() at episode start.
    """

    def reset(self, obs: Dict[str, Any]) -> None: ...

    def act(self, obs: Dict[str, Any]) -> np.ndarray: ...


