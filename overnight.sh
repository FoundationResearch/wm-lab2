#!/usr/bin/env bash
set -euo pipefail

# Three sequential runs (static -> wonly -> aonly).
# Edit any args below as needed; commands run strictly in order.

# Ensure conda env is active for MineDojo.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate mc
fi

mkdir -p logs/overnight
mkdir -p data

if ! command -v zstd >/dev/null 2>&1; then
  echo "[overnight] ERROR: zstd not found. Install it first:" >&2
  echo "  sudo apt-get update && sudo apt-get install -y zstd" >&2
  exit 1
fi

# Change this one word to "beta", "gamma", etc.
RUN_WORD="gamma"

NUM_EPISODES=8192

COMMON_ARGS=(
  --num_episodes "${NUM_EPISODES}" --num_workers 32
  --schedule_mode dynamic --resume
  --xvfb_per_worker --xvfb_display_base 90
  --minedojo_headless auto
  --image_size_hw 352,640
  --fps 25 --max_steps 96 --hold_frames 12 --p_jump 0 --cam_interval 1
  --pitch_min_deg -30 --pitch_max_deg 30
)

run_one() {
  local policy="$1"
  local num="$2"
  local exp="${policy}_${RUN_WORD}${num}"
  local out_root="./data/${exp}"

  python data_collector.py \
    "${COMMON_ARGS[@]}" \
    --policy "${policy}" \
    --out_root "${out_root}" \
    --run_name "${exp}" 2>&1 | tee -a "logs/overnight/${exp}.log"

  # mg postprocess (writes mg/ subdir inside each episode folder)
  python -m wm_lab2.tools.postprocess \
    --root "${out_root}" --recursive --only_ok \
    --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${exp}_mg.log"

  # package (no auto-upload)
  tar -C "${out_root}" --exclude=".minedojo_env_init.lock" \
    --transform="s,^,${exp}/," \
    -cf - . \
    | zstd -T0 -19 -o "./data/${exp}.tar.zst" \
    2>&1 | tee -a "logs/overnight/${exp}_pack.log"
}

run_one wasd12holdrandview 1
run_one wasd12holdrandview 2
run_one wasd12holdrandview 3