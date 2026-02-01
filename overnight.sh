#!/usr/bin/env bash
set -euo pipefail

# Three sequential runs (1 -> 2 -> 3).
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
RUN_WORD="beta"

# Use the first letter of RUN_WORD to namespace output dirs (e.g., beta -> bj1/bj2/bj3).
RUN_LETTER="${RUN_WORD:0:1}"
RUN_TAG="${RUN_LETTER}j"

# Used for log/package filenames (no spaces).
EXP_NAME="${RUN_WORD}jump"
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

# 1 (${RUN_TAG}1)
python data_collector.py \
  "${COMMON_ARGS[@]}" \
  --out_root "./data/wasd12holdrandview${RUN_TAG}1" \
  --run_name "${RUN_WORD} jump1" 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}1.log"

# mg postprocess (writes mg/ subdir inside each episode folder)
python -m wm_lab2.tools.postprocess \
  --root "./data/wasd12holdrandview${RUN_TAG}1" --recursive --only_ok \
  --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}1_mg.log"

# package (no auto-upload)
tar -C "./data/wasd12holdrandview${RUN_TAG}1" --exclude=".minedojo_env_init.lock" \
  --transform="s,^,${EXP_NAME}1/," \
  -cf - . \
  | zstd -T0 -19 -o "./data/${EXP_NAME}1.tar.zst" \
  2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}1_pack.log"

# 2 (${RUN_TAG}2)
python data_collector.py \
  "${COMMON_ARGS[@]}" \
  --out_root "./data/wasd12holdrandview${RUN_TAG}2" \
  --run_name "${RUN_WORD} jump2" 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}2.log"

python -m wm_lab2.tools.postprocess \
  --root "./data/wasd12holdrandview${RUN_TAG}2" --recursive --only_ok \
  --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}2_mg.log"

tar -C "./data/wasd12holdrandview${RUN_TAG}2" --exclude=".minedojo_env_init.lock" \
  --transform="s,^,${EXP_NAME}2/," \
  -cf - . \
  | zstd -T0 -19 -o "./data/${EXP_NAME}2.tar.zst" \
  2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}2_pack.log"

# 3 (${RUN_TAG}3)
python data_collector.py \
  "${COMMON_ARGS[@]}" \
  --out_root "./data/wasd12holdrandview${RUN_TAG}3" \
  --run_name "${RUN_WORD} jump3" 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}3.log"

python -m wm_lab2.tools.postprocess \
  --root "./data/wasd12holdrandview${RUN_TAG}3" --recursive --only_ok \
  --processor mg --out_mode subdir --overwrite 2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}3_mg.log"

tar -C "./data/wasd12holdrandview${RUN_TAG}3" --exclude=".minedojo_env_init.lock" \
  --transform="s,^,${EXP_NAME}3/," \
  -cf - . \
  | zstd -T0 -19 -o "./data/${EXP_NAME}3.tar.zst" \
  2>&1 | tee -a "logs/overnight/${EXP_NAME}_${RUN_TAG}3_pack.log"