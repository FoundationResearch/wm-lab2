from __future__ import annotations

from typing import Dict, List

from wm_lab2.postprocess.actions_json import ActionsJSONProcessor
from wm_lab2.postprocess.base import Processor
from wm_lab2.postprocess.hyw_4average import HYW4AverageProcessor
from wm_lab2.postprocess.mg12average import MG12AverageProcessor


_PROCESSORS: Dict[str, Processor] = {
    "actions_json": ActionsJSONProcessor(),
    "hyw-4average": HYW4AverageProcessor(),
    "mg12average": MG12AverageProcessor(),
}


def list_processors() -> List[str]:
    return sorted(_PROCESSORS.keys())


def get_processor(name: str) -> Processor:
    key = name.strip().lower()
    if key not in _PROCESSORS:
        raise ValueError(f"Unknown processor '{name}'. Available: {', '.join(list_processors())}")
    return _PROCESSORS[key]


