from __future__ import annotations

import json
import os
import traceback
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from subprocess import DEVNULL
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from wm_lab2.inputs.factory import make_policy
from wm_lab2.output.manifest import build_manifest
from wm_lab2.output.writer import write_episode
from wm_lab2.sim.env import EnvSpec, make_env
from wm_lab2.sim.runner import run_episode
from wm_lab2.utils import ensure_dir, utc_timestamp


@dataclass(frozen=True)
class CollectConfig:
    num_episodes: int = 10
    num_workers: int = 1
    max_steps: int = 125
    fps: int = 25
    seed: int = 0
    out_root: str = "./dataset"
    # Optional user-provided tag appended to episode folder names (helps avoid collisions across runs).
    run_name: str = ""
    no_progress: bool = False
    # Env
    task_id: str = "open-ended"
    image_size_hw: tuple[int, int] = (160, 256)
    cam_interval: float = 15.0
    # Minecraft client options (patched via MineDojo's bundled template).
    mc_autojump: bool = True
    # Malmo VideoProducer: 0=first person, 1=third person behind, 2=third person facing.
    mc_video_viewpoint: int = 0
    # Policy
    policy: str = "safe_random"  # safe_random | static | wasd | wasdonly | wonly | aonly | sonly | donly | cam_*12hold | wasd12holdxorcam | ... | custom
    policy_entrypoint: Optional[str] = None  # for custom
    # Policy knobs
    active_dims: Optional[tuple[int, ...]] = None  # for safe_random
    non_stationary_prob: float = 0.95  # for safe_random
    p_jump: float = 0.05  # for wasd
    hold_frames: int = 4  # for wasd4hold
    # Policy knobs (wasd12holdrandview)
    pitch_min_deg: float = -45.0
    pitch_max_deg: float = 45.0
    # Parallel safety: serialize the first MineDojo env.reset() across workers to avoid
    # concurrent ForgeGradle/Malmo cache initialization corruption.
    serialize_env_init: bool = True
    # Parallel scheduling:
    # - "dynamic": workers pull next episode index from a shared queue when ready.
    # - "static": pre-split episodes evenly across workers at startup (legacy behavior).
    schedule_mode: str = "dynamic"
    # Fault tolerance (dynamic schedule only):
    # - If a worker crashes mid-episode, that in-flight episode index is re-queued.
    # - max_episode_retries == 0 means retry forever.
    max_episode_retries: int = 0
    # Resume support (dynamic schedule only):
    # If True, treat `num_episodes` as the target total and only collect missing episode_index
    # under out_root (matching run_name when provided).
    resume: bool = False
    # Xvfb support (headless)
    xvfb_per_worker: bool = False
    xvfb_display_base: int = 90
    # MINEDOJO_HEADLESS handling: "auto"|"0"|"1"
    # - If env var already set, we keep it.
    # - If "1" or "0", we set it explicitly.
    # - If "auto", we set it to "1" when using Xvfb or when DISPLAY is missing.
    minedojo_headless: str = "auto"


def _seed_for_worker(root_seed: int, worker_id: int) -> int:
    # A simple, stable mixing function to keep worker RNG streams apart.
    return int(root_seed) + int(worker_id) * 1_000_003


@contextmanager
def _file_lock(lock_path: str):
    """
    Best-effort cross-process lock (POSIX). If locking is unavailable, this is a no-op.
    """
    try:
        import fcntl  # POSIX only

        ensure_dir(os.path.dirname(lock_path) or ".")
        with open(lock_path, "a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
    except Exception:
        # No-op fallback (e.g., non-POSIX). Parallel init may still race.
        yield


@contextmanager
def _xvfb_display(*, display: int):
    """
    Start an Xvfb server for the lifetime of this context and set DISPLAY=":<display>".
    Requires `Xvfb` installed (Ubuntu package: xvfb).
    """
    import subprocess
    import time

    old_display = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = f":{int(display)}"

    # If Xvfb isn't installed or cannot start, we still proceed (MineDojo may be truly headless).
    proc = None
    try:
        cmd = [
            "Xvfb",
            os.environ["DISPLAY"],
            "-screen",
            "0",
            "1024x768x24",
            "-nolisten",
            "tcp",
        ]
        proc = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
        # Give it a moment to come up.
        time.sleep(0.2)
        yield
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        if old_display is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old_display


def _collect_single_process(cfg: CollectConfig) -> int:
    ensure_dir(cfg.out_root)
    rng = np.random.default_rng(int(cfg.seed))
    run_suffix = f"_{cfg.run_name}" if str(cfg.run_name).strip() else ""

    env = None
    try:
        # If user didn't export MINEDOJO_HEADLESS, decide based on config/environment.
        if "MINEDOJO_HEADLESS" not in os.environ:
            mode = str(cfg.minedojo_headless).strip().lower()
            if mode in ("0", "1"):
                os.environ["MINEDOJO_HEADLESS"] = mode
            elif mode == "auto":
                if os.environ.get("DISPLAY", "") == "":
                    os.environ["MINEDOJO_HEADLESS"] = "1"

        env, nvec, noop = make_env(
            EnvSpec(
                task_id=cfg.task_id,
                image_size_hw=cfg.image_size_hw,
                cam_interval=float(cfg.cam_interval),
                mc_autojump=bool(getattr(cfg, "mc_autojump", True)),
                mc_video_viewpoint=int(getattr(cfg, "mc_video_viewpoint", 0)),
            )
        )
        policy = make_policy(
            name=cfg.policy,
            nvec=nvec,
            noop=noop,
            rng=rng,
            active_dims=cfg.active_dims,
            non_stationary_prob=cfg.non_stationary_prob,
            p_jump=cfg.p_jump,
            hold_frames=cfg.hold_frames,
            pitch_min_deg=float(cfg.pitch_min_deg),
            pitch_max_deg=float(cfg.pitch_max_deg),
            cam_interval_deg=float(cfg.cam_interval),
            entrypoint=cfg.policy_entrypoint,
        )

        outer = range(cfg.num_episodes)
        if not cfg.no_progress:
            outer = tqdm(outer, desc="Episodes")

        for ep_idx in outer:
            ep_stamp = utc_timestamp()
            episode_dir = os.path.join(cfg.out_root, f"episode_{ep_stamp}_{ep_idx:05d}{run_suffix}")
            ensure_dir(episode_dir)

            manifest = build_manifest(
                episode_index=ep_idx,
                timestamp_utc=ep_stamp,
                seed=cfg.seed,
                max_steps=cfg.max_steps,
                fps=cfg.fps,
                task_id=cfg.task_id,
                image_size_hw=cfg.image_size_hw,
                action_nvec=[int(x) for x in nvec.tolist()],
                policy_name=cfg.policy,
            )
            manifest["cam_interval"] = float(cfg.cam_interval)
            manifest["mc_video_viewpoint"] = int(getattr(cfg, "mc_video_viewpoint", 0))
            manifest["num_workers"] = int(cfg.num_workers)
            manifest["worker_id"] = 0
            manifest["worker_seed"] = int(cfg.seed)
            manifest["run_name"] = str(cfg.run_name)
            manifest["pitch_min_deg"] = float(cfg.pitch_min_deg)
            manifest["pitch_max_deg"] = float(cfg.pitch_max_deg)
            # Back-compat keys (kept from original script)
            manifest.setdefault("image_size_hw", [int(cfg.image_size_hw[0]), int(cfg.image_size_hw[1])])
            manifest.setdefault("action_nvec", [int(x) for x in nvec.tolist()])

            with open(os.path.join(episode_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Global warmup for all policies (not recorded):
            # 0: jump, 1..9: W, 10..19: noop
            warmup = 20
            episode = run_episode(env=env, policy=policy, max_steps=cfg.max_steps, warmup_steps=warmup)
            write_episode(out_dir=episode_dir, episode=episode, fps=cfg.fps)

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return 0


def _worker_main(
    cfg: CollectConfig,
    *,
    worker_id: int,
    num_eps: Optional[int] = None,
    global_offset: int = 0,
    work_q=None,
    event_q=None,
    progress_q=None,
) -> None:
    """
    One worker process: owns exactly one MineDojo env (=> one MC instance).
    - Static mode: collects `num_eps` episodes starting at `global_offset`.
    - Dynamic mode: pulls global episode indices from `work_q` until it receives a sentinel `None`.
    """
    ensure_dir(cfg.out_root)
    worker_seed = _seed_for_worker(cfg.seed, worker_id)
    rng = np.random.default_rng(int(worker_seed))

    env = None
    try:
        run_suffix = f"_{cfg.run_name}" if str(cfg.run_name).strip() else ""
        xvfb_ctx = (
            _xvfb_display(display=int(cfg.xvfb_display_base) + int(worker_id))
            if bool(cfg.xvfb_per_worker) and int(cfg.num_workers) > 1
            else nullcontext()
        )
        with xvfb_ctx:
            # Decide MINEDOJO_HEADLESS if not already set by the parent shell.
            if "MINEDOJO_HEADLESS" not in os.environ:
                mode = str(cfg.minedojo_headless).strip().lower()
                if mode in ("0", "1"):
                    os.environ["MINEDOJO_HEADLESS"] = mode
                elif mode == "auto":
                    if bool(cfg.xvfb_per_worker) or os.environ.get("DISPLAY", "") == "":
                        os.environ["MINEDOJO_HEADLESS"] = "1"

            env, nvec, noop = make_env(
                EnvSpec(
                    task_id=cfg.task_id,
                    image_size_hw=cfg.image_size_hw,
                    cam_interval=float(cfg.cam_interval),
                    mc_autojump=bool(getattr(cfg, "mc_autojump", True)),
                    mc_video_viewpoint=int(getattr(cfg, "mc_video_viewpoint", 0)),
                )
            )
            policy = make_policy(
                name=cfg.policy,
                nvec=nvec,
                noop=noop,
                rng=rng,
                active_dims=cfg.active_dims,
                non_stationary_prob=cfg.non_stationary_prob,
                p_jump=cfg.p_jump,
                hold_frames=cfg.hold_frames,
                pitch_min_deg=float(cfg.pitch_min_deg),
                pitch_max_deg=float(cfg.pitch_max_deg),
                cam_interval_deg=float(cfg.cam_interval),
                entrypoint=cfg.policy_entrypoint,
            )

            # Warmup: serialize the very first env.reset() across workers to avoid concurrent
            # initialization of MineDojo/Malmo/ForgeGradle caches (can cause "Corrupted pack file").
            if bool(cfg.serialize_env_init) and int(cfg.num_workers) > 1:
                lock_path = os.path.join(cfg.out_root, ".minedojo_env_init.lock")
                with _file_lock(lock_path):
                    _ = env.reset()

            # One-time init signal (useful for debugging slow startups / stuck workers).
            print(
                f"[collector] worker {int(worker_id)} env ready (pid={os.getpid()}, DISPLAY={os.environ.get('DISPLAY','')})",
                flush=True,
            )

            local_idx = 0
            while True:
                if work_q is None:
                    if num_eps is None:
                        break
                    if local_idx >= int(num_eps):
                        break
                    global_idx = int(global_offset) + int(local_idx)
                else:
                    item = work_q.get()
                    if item is None:
                        # Sentinel: stop this worker.
                        return
                    global_idx = int(item)

                if event_q is not None:
                    try:
                        event_q.put(("start", int(worker_id), int(global_idx)))
                    except Exception:
                        pass

                ep_stamp = utc_timestamp()
                episode_dir = os.path.join(cfg.out_root, f"episode_{ep_stamp}_w{worker_id:02d}_{global_idx:07d}{run_suffix}")
                ensure_dir(episode_dir)

                try:
                    manifest = build_manifest(
                        episode_index=global_idx,
                        timestamp_utc=ep_stamp,
                        seed=int(worker_seed),
                        max_steps=cfg.max_steps,
                        fps=cfg.fps,
                        task_id=cfg.task_id,
                        image_size_hw=cfg.image_size_hw,
                        action_nvec=[int(x) for x in nvec.tolist()],
                        policy_name=cfg.policy,
                    )
                    manifest["cam_interval"] = float(cfg.cam_interval)
                    manifest["mc_video_viewpoint"] = int(getattr(cfg, "mc_video_viewpoint", 0))
                    manifest["num_workers"] = int(cfg.num_workers)
                    manifest["worker_id"] = int(worker_id)
                    manifest["worker_seed"] = int(worker_seed)
                    manifest["root_seed"] = int(cfg.seed)
                    manifest["episode_index_local"] = int(local_idx)
                    manifest["pitch_min_deg"] = float(cfg.pitch_min_deg)
                    manifest["pitch_max_deg"] = float(cfg.pitch_max_deg)
                    manifest["xvfb_per_worker"] = bool(cfg.xvfb_per_worker)
                    manifest["display"] = os.environ.get("DISPLAY", "")
                    manifest["run_name"] = str(cfg.run_name)
                    manifest["schedule_mode"] = str(getattr(cfg, "schedule_mode", "dynamic"))
                    # Back-compat keys (kept from original script)
                    manifest.setdefault("image_size_hw", [int(cfg.image_size_hw[0]), int(cfg.image_size_hw[1])])
                    manifest.setdefault("action_nvec", [int(x) for x in nvec.tolist()])

                    with open(os.path.join(episode_dir, "manifest.json"), "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2)

                    # Global warmup for all policies (not recorded):
                    # 0: jump, 1..9: W, 10..19: noop
                    warmup = 20
                    episode = run_episode(env=env, policy=policy, max_steps=cfg.max_steps, warmup_steps=warmup)
                    write_episode(out_dir=episode_dir, episode=episode, fps=cfg.fps)
                except BaseException as e:
                    if event_q is not None:
                        try:
                            event_q.put(("fail", int(worker_id), int(global_idx), repr(e)))
                        except Exception:
                            pass
                    # Most MineDojo failures (socket timeout, instance crash) leave the env in a bad state.
                    # Crash the worker so the parent can re-queue the episode and continue with others.
                    raise
                else:
                    if event_q is not None:
                        try:
                            event_q.put(("done", int(worker_id), int(global_idx)))
                        except Exception:
                            pass
                    if progress_q is not None:
                        progress_q.put(1)
                finally:
                    local_idx += 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _collect_parallel(cfg: CollectConfig) -> int:
    import multiprocessing as mp

    num_workers = max(1, int(cfg.num_workers))
    if num_workers <= 1:
        return _collect_single_process(cfg)

    ctx = mp.get_context("spawn")
    schedule_mode = str(getattr(cfg, "schedule_mode", "dynamic")).strip().lower()
    if schedule_mode not in ("dynamic", "static"):
        raise ValueError(f"Unknown schedule_mode={schedule_mode!r} (expected 'dynamic' or 'static').")

    procs: dict[int, Any] = {}

    def _start_worker(wid: int, *, work_q=None, event_q=None, progress_q=None, num_eps: Optional[int] = None, offset: int = 0):
        p = ctx.Process(
            target=_worker_main,
            kwargs=dict(
                cfg=cfg,
                worker_id=int(wid),
                num_eps=(int(num_eps) if num_eps is not None else None),
                global_offset=int(offset),
                work_q=work_q,
                event_q=event_q,
                progress_q=progress_q,
            ),
            daemon=False,
        )
        p.start()
        procs[int(wid)] = p

    def _terminate_all() -> None:
        for p in procs.values():
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in procs.values():
            try:
                p.join(timeout=5)
            except Exception:
                pass

    # --- Dynamic schedule: shared queue + crash re-queue ---
    if schedule_mode == "dynamic":
        total = int(cfg.num_episodes)
        work_q = ctx.Queue()
        event_q = ctx.Queue()

        def _load_existing_episode_indices() -> set[int]:
            existing: set[int] = set()
            try:
                if not bool(getattr(cfg, "resume", False)):
                    return existing
                out_root = str(cfg.out_root)
                run_name = str(getattr(cfg, "run_name", "")).strip()
                if not os.path.isdir(out_root):
                    return existing
                for name in os.listdir(out_root):
                    if not name.startswith("episode_"):
                        continue
                    mpath = os.path.join(out_root, name, "manifest.json")
                    if not os.path.isfile(mpath):
                        continue
                    try:
                        with open(mpath, "r", encoding="utf-8") as f:
                            m = json.load(f)
                        if run_name:
                            if str(m.get("run_name", "")).strip() != run_name:
                                continue
                        idx = m.get("episode_index", None)
                        if idx is None:
                            continue
                        idx_i = int(idx)
                        if 0 <= idx_i < total:
                            existing.add(idx_i)
                    except Exception:
                        # Ignore corrupted/partial manifests.
                        continue
            except Exception:
                return existing
            return existing

        existing = _load_existing_episode_indices()
        completed: set[int] = set(existing)

        remaining_indices = [i for i in range(total) if i not in existing]
        for ep_idx in remaining_indices:
            work_q.put(int(ep_idx))

        for wid in range(num_workers):
            _start_worker(int(wid), work_q=work_q, event_q=event_q)

        in_flight: dict[int, Optional[int]] = {int(wid): None for wid in range(num_workers)}
        permanently_failed: dict[int, str] = {}
        attempts = defaultdict(int)
        max_retries = int(getattr(cfg, "max_episode_retries", 0))

        def _maybe_requeue(ep_idx: int, why: str) -> None:
            if int(ep_idx) in completed:
                return
            attempts[int(ep_idx)] += 1
            if max_retries > 0 and attempts[int(ep_idx)] > max_retries:
                # Give up: mark as completed so the run can finish, and surface an error at the end.
                completed.add(int(ep_idx))
                permanently_failed[int(ep_idx)] = str(why)
                return
            work_q.put(int(ep_idx))

        try:
            if cfg.no_progress:
                # Still need to consume events and monitor crashed workers.
                while len(completed) < total:
                    try:
                        ev = event_q.get(timeout=1.0)
                    except Exception:
                        ev = None

                    if ev is not None:
                        kind = ev[0]
                        if kind == "start":
                            _, wid, ep = ev
                            in_flight[int(wid)] = int(ep)
                        elif kind == "done":
                            _, wid, ep = ev
                            in_flight[int(wid)] = None
                            completed.add(int(ep))
                        elif kind == "fail":
                            _, wid, ep, err = ev
                            in_flight[int(wid)] = None
                            _maybe_requeue(int(ep), str(err))

                    # Detect crashed workers and re-queue their in-flight episode.
                    for wid, p in list(procs.items()):
                        if p.exitcode is not None and p.exitcode != 0:
                            ep = in_flight.get(int(wid))
                            if ep is not None:
                                in_flight[int(wid)] = None
                                _maybe_requeue(int(ep), f"worker {wid} crashed (exitcode={p.exitcode})")
                            procs.pop(int(wid), None)
            else:
                bar_total = max(0, total - len(existing))
                with tqdm(total=bar_total, desc=f"Episodes (parallel x{num_workers}, dynamic)") as bar:
                    while len(completed) < total:
                        try:
                            ev = event_q.get(timeout=1.0)
                        except Exception:
                            ev = None

                        if ev is not None:
                            kind = ev[0]
                            if kind == "start":
                                _, wid, ep = ev
                                in_flight[int(wid)] = int(ep)
                            elif kind == "done":
                                _, wid, ep = ev
                                in_flight[int(wid)] = None
                                if int(ep) not in completed:
                                    completed.add(int(ep))
                                    bar.update(1)
                            elif kind == "fail":
                                _, wid, ep, err = ev
                                in_flight[int(wid)] = None
                                try:
                                    bar.write(f"[collector] worker {wid} episode {ep} failed: {err} (re-queue)")
                                except Exception:
                                    pass
                                _maybe_requeue(int(ep), str(err))

                        # Detect crashed workers and re-queue their in-flight episode.
                        for wid, p in list(procs.items()):
                            if p.exitcode is not None and p.exitcode != 0:
                                ep = in_flight.get(int(wid))
                                if ep is not None:
                                    in_flight[int(wid)] = None
                                    try:
                                        bar.write(
                                            f"[collector] worker {wid} crashed (exitcode={p.exitcode}); re-queue episode {ep}"
                                        )
                                    except Exception:
                                        pass
                                    _maybe_requeue(int(ep), f"worker {wid} crashed (exitcode={p.exitcode})")
                                procs.pop(int(wid), None)

                        if not procs and len(completed) < total:
                            raise RuntimeError(
                                f"All workers exited but {total - len(completed)} episodes remain unfinished."
                            )
        except BaseException:
            _terminate_all()
            raise
        finally:
            # Ask remaining live workers to stop.
            for _ in range(len(procs)):
                try:
                    work_q.put(None)
                except Exception:
                    pass
            for p in procs.values():
                try:
                    p.join()
                except Exception:
                    pass
            try:
                event_q.close()
                event_q.join_thread()
            except Exception:
                pass
            try:
                work_q.close()
                work_q.join_thread()
            except Exception:
                pass

        if permanently_failed:
            # Keep the run going but surface a clear summary at the end.
            keys = sorted(permanently_failed.keys())
            msg = (
                f"{len(keys)} episode(s) failed permanently after max_episode_retries={max_retries}: "
                f"{keys[:20]}{' ...' if len(keys) > 20 else ''}"
            )
            raise RuntimeError(msg)
        return 0

    # --- Static schedule: legacy pre-split ---
    # Split episodes across workers as evenly as possible.
    n = int(cfg.num_episodes)
    base = n // num_workers
    rem = n % num_workers
    counts = [base + (1 if i < rem else 0) for i in range(num_workers)]
    offsets = []
    cur = 0
    for c in counts:
        offsets.append(cur)
        cur += c

    progress_q = ctx.Queue()
    for wid in range(num_workers):
        c = int(counts[wid])
        if c <= 0:
            continue
        _start_worker(int(wid), num_eps=int(c), offset=int(offsets[wid]), progress_q=progress_q)

    try:
        # Progress: best-effort; if disabled, just block until join.
        total = int(cfg.num_episodes)
        if cfg.no_progress:
            for p in procs.values():
                p.join()
        else:
            with tqdm(total=total, desc=f"Episodes (parallel x{num_workers})") as bar:
                done = 0
                while done < total:
                    try:
                        inc = progress_q.get(timeout=1.0)
                        done += int(inc)
                        bar.update(int(inc))
                    except Exception:
                        # Timeout: check for early worker exits.
                        pass
                    if any((p.exitcode is not None and p.exitcode != 0) for p in procs.values()):
                        break
                for p in procs.values():
                    p.join()
    except BaseException:
        # IMPORTANT: On Ctrl+C, terminate workers so file locks (e.g. env init lock) are released.
        _terminate_all()
        raise
    finally:
        try:
            progress_q.close()
            progress_q.join_thread()
        except Exception:
            pass

    bad = [p for p in procs.values() if p.exitcode not in (0, None)]
    if bad:
        # Surface a minimal diagnostic; detailed tracebacks should be visible in stderr logs.
        raise RuntimeError(f"{len(bad)}/{len(procs)} collector workers failed (exit codes: {[p.exitcode for p in bad]}).")

    return 0


def collect(cfg: CollectConfig) -> int:
    try:
        if int(cfg.num_workers) <= 1:
            return _collect_single_process(cfg)
        return _collect_parallel(cfg)
    except Exception:
        # Keep a readable traceback for CLI users.
        traceback.print_exc()
        return 1


