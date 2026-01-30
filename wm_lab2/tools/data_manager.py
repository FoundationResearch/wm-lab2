from __future__ import annotations

import argparse
import cmd
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from wm_lab2.data_index import EpisodeRecord, EpisodeStatus, scan_dataset


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n_f = n / 1024.0
        if n_f < 1024.0:
            return f"{n_f:.1f}{unit}"
        n = int(n_f)
    return f"{n}TB+"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += int(p.stat().st_size)
                except Exception:
                    pass
    except Exception:
        pass
    return total


@dataclass
class ManagerState:
    root: Path
    recursive: bool
    empty_video_bytes: int
    episodes: List[EpisodeRecord]
    by_id: Dict[str, EpisodeRecord]


class DataManagerShell(cmd.Cmd):
    intro = "wm_lab2 data manager. Type 'help' for commands.\n"
    prompt = "(wm_data) "

    def __init__(self, state: ManagerState):
        super().__init__()
        self.state = state

    # ---- helpers ----
    def _rescan(self) -> None:
        eps = scan_dataset(self.state.root, recursive=self.state.recursive, empty_video_bytes=self.state.empty_video_bytes)
        self.state.episodes = eps
        self.state.by_id = {e.episode_id: e for e in eps}

    def _get(self, ep_id: str) -> Optional[EpisodeRecord]:
        return self.state.by_id.get(ep_id)

    def _resolve(self, token: str) -> Optional[EpisodeRecord]:
        # exact id
        if token in self.state.by_id:
            return self.state.by_id[token]
        # prefix match
        matches = [e for e in self.state.episodes if e.episode_id.startswith(token)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _confirm(self, msg: str) -> bool:
        ans = input(f"{msg} [y/N] ").strip().lower()
        return ans in ("y", "yes")

    # ---- commands ----
    def do_rescan(self, arg: str) -> None:
        "Rescan dataset root and refresh episode statuses."
        self._rescan()
        print(f"scanned {len(self.state.episodes)} episodes under {self.state.root}")

    def do_ls(self, arg: str) -> None:
        "List episodes. Usage: ls [ok|empty|failed]  (also shows category if present)"
        a = arg.strip().lower()
        status = a if a in (s.value for s in EpisodeStatus) else None
        rows = []
        for e in self.state.episodes:
            if status and e.status.value != status:
                continue
            T = "-" if e.num_steps is None else str(e.num_steps)
            D = "-" if e.action_dim is None else str(e.action_dim)
            cat = e.category or "-"
            rows.append((cat, e.episode_id, e.status.value, T, D, e.reason))
        for (cat, ep_id, st, T, D, reason) in rows:
            print(f"{cat:12} {ep_id}  [{st}]  T={T} D={D}  {reason}")
        if not rows:
            print("(no matches)")

    def complete_ls(self, text: str, line: str, begidx: int, endidx: int):
        return self._complete_from(text, [s.value for s in EpisodeStatus])

    def do_show(self, arg: str) -> None:
        "Show one episode details. Usage: show <episode_id_or_prefix>"
        tok = arg.strip()
        if not tok:
            print("usage: show <episode_id_or_prefix>")
            return
        e = self._resolve(tok)
        if e is None:
            print(f"not found / ambiguous: {tok}")
            return
        size = _human_bytes(_dir_size(e.paths.root))
        print(f"episode: {e.episode_id}")
        print(f"category: {e.category or '-'}")
        print(f"status : {e.status.value} ({e.reason})")
        print(f"path   : {e.paths.root}")
        print(f"size   : {size}")
        print(f"steps  : {e.num_steps}")
        print(f"action : D={e.action_dim}")
        if e.manifest:
            print("manifest keys:", ", ".join(sorted(e.manifest.keys())))
            print("policy:", e.manifest.get("policy"))
            print("task_id:", e.manifest.get("task_id"))

    def complete_show(self, text: str, line: str, begidx: int, endidx: int):
        return self._complete_episode_id(text)

    def do_rm(self, arg: str) -> None:
        """
        Delete episodes.

        Usage:
          rm <episode_id_or_prefix>
          rm <ok|empty|failed>
          rm all
          rm all <ok|empty|failed>
        """
        parts = shlex.split(arg)
        if not parts:
            print("usage: rm <episode_id_or_prefix> | rm <ok|empty|failed> | rm all [ok|empty|failed]")
            return

        status_words = {s.value for s in EpisodeStatus}

        # rm all [status]
        if parts[0].lower() == "all":
            if len(parts) == 1:
                victims = list(self.state.episodes)
                label = "ALL episodes"
            elif len(parts) == 2 and parts[1].lower() in status_words:
                st = EpisodeStatus(parts[1].lower())
                victims = [e for e in self.state.episodes if e.status == st]
                label = f"ALL {st.value} episodes"
            else:
                print("usage: rm all [ok|empty|failed]")
                return

            print(f"matches: {len(victims)}")
            if not victims:
                return
            if not self._confirm(f"Delete {label}?"):
                print("cancelled")
                return
            for e in victims:
                shutil.rmtree(e.paths.root, ignore_errors=True)
            print("done")
            self._rescan()
            return

        # rm <status>
        if len(parts) == 1 and parts[0].lower() in status_words:
            st = EpisodeStatus(parts[0].lower())
            victims = [e for e in self.state.episodes if e.status == st]
            print(f"matches: {len(victims)}")
            if not victims:
                return
            if not self._confirm(f"Delete ALL {st.value} episodes?"):
                print("cancelled")
                return
            for e in victims:
                shutil.rmtree(e.paths.root, ignore_errors=True)
            print("done")
            self._rescan()
            return

        # rm <episode_id_or_prefix>
        if len(parts) != 1:
            print("usage: rm <episode_id_or_prefix> | rm <ok|empty|failed> | rm all [ok|empty|failed]")
            return

        tok = parts[0]
        e = self._resolve(tok)
        if e is None:
            print(f"not found / ambiguous: {tok}")
            return
        if not self._confirm(f"Delete {e.episode_id} at {e.paths.root}?"):
            print("cancelled")
            return
        shutil.rmtree(e.paths.root)
        print(f"deleted {e.episode_id}")
        self._rescan()

    def complete_rm(self, text: str, line: str, begidx: int, endidx: int):
        # rm <...>
        parts = shlex.split(line[:begidx])
        # parts includes "rm" and already-complete args before current token
        args = parts[1:] if parts and parts[0] == "rm" else parts

        status_words = [s.value for s in EpisodeStatus]
        if len(args) == 0:
            return self._complete_from(text, ["all"] + status_words + self._episode_ids())
        if len(args) == 1 and args[0].lower() == "all":
            return self._complete_from(text, status_words)
        return []

    def do_mv(self, arg: str) -> None:
        """
        Move an episode into a category folder under root.
        Usage: mv <episode_id_or_prefix> <category>

        Result path:
          <root>/<category>/<episode_id>/
        """
        parts = shlex.split(arg)
        if len(parts) != 2:
            print("usage: mv <episode_id_or_prefix> <category>")
            return
        tok, category = parts
        e = self._resolve(tok)
        if e is None:
            print(f"not found / ambiguous: {tok}")
            return
        dst_parent = self.state.root / category
        dst = dst_parent / e.episode_id
        dst_parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"destination exists: {dst}")
            return
        shutil.move(str(e.paths.root), str(dst))
        print(f"moved to {dst}")
        self._rescan()

    def complete_mv(self, text: str, line: str, begidx: int, endidx: int):
        parts = shlex.split(line[:begidx])
        args = parts[1:] if parts and parts[0] == "mv" else parts
        if len(args) == 0:
            return self._complete_episode_id(text)
        if len(args) == 1:
            # complete category
            return self._complete_from(text, self._category_suggestions())
        return []

    def do_open(self, arg: str) -> None:
        "Open an episode folder in Finder (macOS). Usage: open <episode_id_or_prefix>"
        tok = arg.strip()
        if not tok:
            print("usage: open <episode_id_or_prefix>")
            return
        e = self._resolve(tok)
        if e is None:
            print(f"not found / ambiguous: {tok}")
            return
        # Best-effort; ok if not on macOS.
        os.system(f"open {shlex.quote(str(e.paths.root))}")

    def complete_open(self, text: str, line: str, begidx: int, endidx: int):
        return self._complete_episode_id(text)

    def do_quit(self, arg: str) -> bool:
        "Quit."
        return True

    def do_exit(self, arg: str) -> bool:
        "Quit."
        return True

    # ---- completion helpers ----
    def _episode_ids(self) -> List[str]:
        return [e.episode_id for e in self.state.episodes]

    def _complete_from(self, text: str, candidates: Sequence[str]) -> List[str]:
        t = text or ""
        return sorted([c for c in candidates if c.startswith(t)])

    def _complete_episode_id(self, text: str) -> List[str]:
        return self._complete_from(text, self._episode_ids())

    def _category_suggestions(self) -> List[str]:
        # Suggest existing directories under root (excluding episode_*), plus common buckets.
        out: List[str] = ["ok", "empty", "failed"]
        try:
            for p in self.state.root.iterdir():
                if p.is_dir() and not p.name.startswith("episode_") and not p.name.startswith("."):
                    out.append(p.name)
        except Exception:
            pass
        # de-dup
        return sorted(set(out))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Interactive dataset manager for MineDojo episodes.")
    p.add_argument("--root", type=str, default="./data", help="Dataset root containing episode_* folders.")
    p.add_argument("--recursive", action="store_true", help="Also scan nested episode_* folders.")
    p.add_argument("--empty_video_bytes", type=int, default=1024, help="video.mp4 <= this is treated as empty.")
    args = p.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).expanduser().resolve()
    eps = scan_dataset(root, recursive=bool(args.recursive), empty_video_bytes=int(args.empty_video_bytes))
    state = ManagerState(
        root=root,
        recursive=bool(args.recursive),
        empty_video_bytes=int(args.empty_video_bytes),
        episodes=eps,
        by_id={e.episode_id: e for e in eps},
    )
    DataManagerShell(state).cmdloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


