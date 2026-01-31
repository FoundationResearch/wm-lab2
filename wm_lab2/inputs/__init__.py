from wm_lab2.inputs.base import Policy
from wm_lab2.inputs.factory import make_policy
from wm_lab2.inputs.safe_random import SafeRandomPolicy
from wm_lab2.inputs.wasd import WASDOnlyPolicy
from wm_lab2.inputs.wasd4hold import WASD4HoldPolicy
from wm_lab2.inputs.wasd12holdrandview import WASD12HoldRandViewPolicy

__all__ = ["Policy", "make_policy", "SafeRandomPolicy", "WASDOnlyPolicy", "WASD4HoldPolicy", "WASD12HoldRandViewPolicy"]


