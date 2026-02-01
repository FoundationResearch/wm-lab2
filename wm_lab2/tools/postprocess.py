from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tqdm import tqdm

from wm_lab2.data_index import EpisodeStatus, scan_dataset
from wm_lab2.postprocess.registry import get_processor, list_processors


def _safe_read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Post-process MineDojo episodes (npz -> derived formats).")
    p.add_argument("--root", type=str, default="./data", help="Dataset root.")
    p.add_argument("--recursive", action="store_true", help="Also scan nested episode_* folders.")
    p.add_argument(
        "--processor",
        type=str,
        default="actions_json",
        help=f"Processor name. Available: {', '.join(list_processors())}",
    )
    p.add_argument("--only_ok", action="store_true", help="Only process episodes classified as ok.")
    p.add_argument(
        "--out_mode",
        type=str,
        default="inplace",
        choices=["inplace", "subdir"],
        help="Where to write outputs: inplace (episode dir) or subdir (episode_dir/<processor_name>/).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing derived outputs.")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bar.")

    args = p.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).expanduser().resolve()
    proc = get_processor(args.processor)
    episodes = scan_dataset(root, recursive=bool(args.recursive))

    todo = []
    for ep in episodes:
        if args.only_ok and ep.status != EpisodeStatus.ok:
            continue
        if not ep.paths.npz.exists() or not ep.paths.manifest.exists():
            continue
        todo.append(ep)

    n_total = len(todo)
    n_done = 0

    it = todo
    if not bool(args.no_progress):
        it = tqdm(todo, desc=f"Postprocess ({proc.name})", unit="ep")

    for ep in it:
        manifest = _safe_read_json(ep.paths.manifest)
        out_dir = ep.paths.root if args.out_mode == "inplace" else (ep.paths.root / proc.name)
        proc.process(
            episode_dir=ep.paths.root,
            manifest=manifest,
            npz_path=ep.paths.npz,
            out_dir=out_dir,
            overwrite=bool(args.overwrite),
        )
        n_done += 1

    print(f"processed {n_done}/{n_total} episodes with processor='{proc.name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


