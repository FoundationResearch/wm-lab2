from __future__ import annotations

from typing import Any, Dict, Protocol

from pathlib import Path


class Processor(Protocol):
    """
    Post-processing plugin interface.

    A processor reads episode artifacts (manifest.json, data.npz, video.mp4)
    and writes derived outputs.
    """

    name: str

    def process(
        self,
        *,
        episode_dir: Path,
        manifest: Dict[str, Any],
        npz_path: Path,
        out_dir: Path,
        overwrite: bool,
    ) -> None: ...


