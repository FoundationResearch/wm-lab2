#!/usr/bin/env python3
"""
Upload large dataset archives to Hugging Face (dataset repo).

Two modes:

1) Proxy mode (mirrors `huggingface-cli upload` arguments):

   python upload_to_hf.py upload <repo_id> <local_path> \\
     --repo-type dataset \\
     --path-in-repo some/path/file.tar.zst \\
     --commit-message "..."

2) Convenience mode (upload alpha/beta archives with defaults):

   python upload_to_hf.py

Auth:
- Prefer environment variable: HF_TOKEN (or HUGGINGFACE_HUB_TOKEN)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _get_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token and str(cli_token).strip():
        return str(cli_token).strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{n} B"


def _upload_one(
    *,
    repo_id: str,
    repo_type: str,
    revision: Optional[str],
    token: Optional[str],
    local_path: Path,
    path_in_repo: str,
    commit_message: str,
    overwrite: bool,
    show_progress: bool,
) -> None:
    from huggingface_hub import HfApi, upload_file

    if not local_path.exists():
        raise FileNotFoundError(f"Missing local file: {local_path}")
    if not local_path.is_file():
        raise RuntimeError(f"Not a file: {local_path}")

    api = HfApi(token=token)
    exists = False
    try:
        exists = bool(
            api.file_exists(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type=repo_type,
                revision=revision,
            )
        )
    except Exception:
        # If we can't check existence (network / permissions), still try upload.
        exists = False

    size = local_path.stat().st_size
    print(f"- local: {local_path} ({_human_bytes(size)})")
    print(f"- repo:  {repo_id} ({repo_type}) :: {path_in_repo}")

    if exists and not overwrite:
        print("  -> already exists on Hub; skip (use --overwrite to re-upload)")
        return

    # Explicit progress bar (works even when huggingface-cli isn't available).
    # If tqdm is not installed or output isn't a TTY, fall back to path upload.
    use_tqdm = bool(show_progress) and sys.stderr.isatty()
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore

            with local_path.open("rb") as f:
                wrapped = tqdm.wrapattr(
                    f,
                    "read",
                    total=size,
                    desc=f"upload {local_path.name}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
                upload_file(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                    path_or_fileobj=wrapped,
                    path_in_repo=path_in_repo,
                    commit_message=commit_message,
                )
        except Exception:
            upload_file(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=commit_message,
            )
    else:
        upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=commit_message,
        )
    print("  -> upload done")


def _parse_proxy_upload(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="upload_to_hf.py upload",
        description="Proxy for `huggingface-cli upload` (implemented via huggingface_hub Python API).",
    )
    p.add_argument("repo_id", type=str, help="e.g. alexzms/FastvideoWorldModel-MC")
    p.add_argument("local_path", type=str, help="Local file to upload")
    p.add_argument("--repo-type", "--repo_type", dest="repo_type", type=str, default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--path-in-repo", "--path_in_repo", dest="path_in_repo", type=str, required=True)
    p.add_argument("--commit-message", "--commit_message", dest="commit_message", type=str, default="Upload file")
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--token", type=str, default=None, help="HF token (or set env HF_TOKEN).")
    p.add_argument("--overwrite", action="store_true", help="Re-upload even if file exists on Hub.")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    return p.parse_args(argv)


def _parse_alpha_beta(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload alpha/beta archives to Hugging Face dataset repo.")
    p.add_argument("--repo_id", type=str, default="alexzms/FastvideoWorldModel-MC")
    p.add_argument("--repo_type", type=str, default="dataset", choices=["dataset", "model", "space"])
    p.add_argument("--revision", type=str, default=None, help="Branch/tag/commit. Default: repo default branch.")
    p.add_argument("--token", type=str, default=None, help="HF token (or set env HF_TOKEN).")
    p.add_argument("--path_prefix", type=str, default="wasd12holdrandview-96frame", help="Folder in repo.")

    p.add_argument("--alpha", type=str, default="data/wasd12holdrandview_alpha.tar.zst", help="Local alpha archive path.")
    p.add_argument("--beta", type=str, default="data/wasd12holdrandview_beta.tar.zst", help="Local beta archive path.")
    p.add_argument("--skip_alpha", action="store_true")
    p.add_argument("--skip_beta", action="store_true")

    p.add_argument("--overwrite", action="store_true", help="Re-upload even if file exists on Hub.")
    p.add_argument("--commit_message", type=str, default="Upload archives", help="Commit message for uploads.")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bar.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "alpha_beta"
    if argv[:1] == ["upload"]:
        mode = "upload"
        argv = argv[1:]

    try:
        if mode == "upload":
            args = _parse_proxy_upload(argv)
            token = _get_token(args.token)
            if not token:
                print("ERROR: Missing HF token. Set env HF_TOKEN='hf_...' or pass --token.", file=sys.stderr)
                return 2
            _upload_one(
                repo_id=str(args.repo_id),
                repo_type=str(args.repo_type),
                revision=args.revision,
                token=token,
                local_path=Path(args.local_path).expanduser().resolve(),
                path_in_repo=str(args.path_in_repo),
                commit_message=str(args.commit_message),
                overwrite=bool(args.overwrite),
                show_progress=not bool(args.no_progress),
            )
            return 0

        # default: alpha/beta convenience mode
        args = _parse_alpha_beta(argv)
        token = _get_token(args.token)
        if not token:
            print("ERROR: Missing HF token. Set env HF_TOKEN='hf_...' or pass --token.", file=sys.stderr)
            return 2

        prefix = str(args.path_prefix).strip().strip("/")
        if prefix == "":
            print("ERROR: --path_prefix cannot be empty", file=sys.stderr)
            return 2

        alpha_path = Path(args.alpha).expanduser().resolve()
        beta_path = Path(args.beta).expanduser().resolve()

        if not bool(args.skip_alpha):
            _upload_one(
                repo_id=str(args.repo_id),
                repo_type=str(args.repo_type),
                revision=args.revision,
                token=token,
                local_path=alpha_path,
                path_in_repo=f"{prefix}/alpha.tar.zst",
                commit_message=str(args.commit_message) + " (alpha)",
                overwrite=bool(args.overwrite),
                show_progress=not bool(args.no_progress),
            )
        if not bool(args.skip_beta):
            _upload_one(
                repo_id=str(args.repo_id),
                repo_type=str(args.repo_type),
                revision=args.revision,
                token=token,
                local_path=beta_path,
                path_in_repo=f"{prefix}/beta.tar.zst",
                commit_message=str(args.commit_message) + " (beta)",
                overwrite=bool(args.overwrite),
                show_progress=not bool(args.no_progress),
            )
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


