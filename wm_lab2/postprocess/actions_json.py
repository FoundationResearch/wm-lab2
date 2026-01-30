from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from wm_lab2.postprocess.base import Processor


@dataclass
class ActionsJSONProcessor(Processor):
    """
    Export `data.npz` actions to `actions.json`.

    Output schema:
    {
      "episode_id": "...",
      "action_nvec": [...],   # if present in manifest
      "actions": [[...], ...] # length T
    }
    """

    name: str = "actions_json"
    output_filename: str = "actions.json"

    def process(
        self,
        *,
        episode_dir: Path,
        manifest: Dict[str, Any],
        npz_path: Path,
        out_dir: Path,
        overwrite: bool,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / self.output_filename
        if out_path.exists() and not overwrite:
            return

        with np.load(npz_path, allow_pickle=False) as z:
            actions = np.asarray(z["actions"], dtype=np.int64)

        payload: Dict[str, Any] = {
            "episode_id": episode_dir.name,
            "action_nvec": manifest.get("action_nvec"),
            "actions": actions.tolist(),
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


