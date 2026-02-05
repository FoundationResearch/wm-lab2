#!/usr/bin/env bash
set -euo pipefail

# Three sequential runs (7 -> 8 -> 9).
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
RUN_WORD="alpha"

# Use the first letter of RUN_WORD to namespace output dirs (e.g., alpha -> aj7/aj8/aj9).
RUN_LETTER="${RUN_WORD:0:1}"
RUN_TAG="${RUN_LETTER}j"

# Used for log/package filenames (no spaces).
EXP_NAME="${RUN_WORD}jumpf"
POLICY="wasd12holdrandview"
COMMON_ARGS=(
  --policy "${POLICY}" --num_episodes 8192 --num_workers 32
  --schedule_mode dynamic --resume
  --xvfb_per_worker --xvfb_display_base 90
  --minedojo_headless auto
  --image_size_hw 352,640
  --fps 25 --max_steps 96 --hold_frames 12 --p_jump 0 --cam_interval 1
  --pitch_min_deg -30 --pitch_max_deg 30
)

# 7 (${RUN_TAG}7)
python data_collector.py \
  "${COMMON_ARGS[@]}" \
  --out_root "./data/wasd12holdrandview${RUN_TAG}7" \
  --run_name "${RUN_WORD} jumpf7" 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}7.log"

# mg postprocess (writes mg/ subdir inside each episode folder)
python -m wm_lab2.tools.postprocess \
  --root "./data/wasd12holdrandview${RUN_TAG}7" --recursive --only_ok \
  --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}7_mg.log"

# package (no auto-upload)
tar -C "./data/wasd12holdrandview${RUN_TAG}7" --exclude=".minedojo_env_init.lock" \
  --transform="s,^,${EXP_NAME}7/," \
  -cf - . \
  | zstd -T0 -19 -o "./data/${EXP_NAME}7.tar.zst" \
  2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}7_pack.log"

# 8 (${RUN_TAG}8)
python data_collector.py \
  "${COMMON_ARGS[@]}" \
  --out_root "./data/wasd12holdrandview${RUN_TAG}8" \
  --run_name "${RUN_WORD} jumpf8" 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}8.log"

python -m wm_lab2.tools.postprocess \
  --root "./data/wasd12holdrandview${RUN_TAG}8" --recursive --only_ok \
  --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}8_mg.log"

tar -C "./data/wasd12holdrandview${RUN_TAG}8" --exclude=".minedojo_env_init.lock" \
  --transform="s,^,${EXP_NAME}8/," \
  -cf - . \
  | zstd -T0 -19 -o "./data/${EXP_NAME}8.tar.zst" \
  2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}8_pack.log"

# 9 (${RUN_TAG}9)
python data_collector.py \
  "${COMMON_ARGS[@]}" \
  --out_root "./data/wasd12holdrandview${RUN_TAG}9" \
  --run_name "${RUN_WORD} jumpf9" 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}9.log"

python -m wm_lab2.tools.postprocess \
  --root "./data/wasd12holdrandview${RUN_TAG}9" --recursive --only_ok \
  --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}9_mg.log"

tar -C "./data/wasd12holdrandview${RUN_TAG}9" --exclude=".minedojo_env_init.lock" \
  --transform="s,^,${EXP_NAME}9/," \
  -cf - . \
  | zstd -T0 -19 -o "./data/${EXP_NAME}9.tar.zst" \
  2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}9_pack.log"