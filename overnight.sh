#!/usr/bin/env bash
set -e

# Three sequential runs (beta3 -> beta4 -> beta5).
# Edit any args below as needed; commands run strictly in order.

mkdir -p logs/overnight

# alpha3
python data_collector.py \
  --policy wasd12holdrandview --num_episodes 8192 --num_workers 32 \
  --schedule_mode dynamic --resume \
  --xvfb_per_worker --xvfb_display_base 90 \
  --minedojo_headless auto \
  --image_size_hw 352,640 --out_root ./data/wasd12holdrandviewa3 \
  --fps 25 --max_steps 96 --hold_frames 12 --p_jump 0 --cam_interval 1 \
  --pitch_min_deg -30 --pitch_max_deg 30 \
  --run_name "alpha3" 2>&1 | tee -a logs/overnight/alpha3.log

# alpha4
python data_collector.py \
  --policy wasd12holdrandview --num_episodes 8192 --num_workers 32 \
  --schedule_mode dynamic --resume \
  --xvfb_per_worker --xvfb_display_base 90 \
  --minedojo_headless auto \
  --image_size_hw 352,640 --out_root ./data/wasd12holdrandviewa4 \
  --fps 25 --max_steps 96 --hold_frames 12 --p_jump 0 --cam_interval 1 \
  --pitch_min_deg -30 --pitch_max_deg 30 \
  --run_name "alpha4" 2>&1 | tee -a logs/overnight/alpha4.log

# alpha5
python data_collector.py \
  --policy wasd12holdrandview --num_episodes 8192 --num_workers 32 \
  --schedule_mode dynamic --resume \
  --xvfb_per_worker --xvfb_display_base 90 \
  --minedojo_headless auto \
  --image_size_hw 352,640 --out_root ./data/wasd12holdrandviewa5 \
  --fps 25 --max_steps 96 --hold_frames 12 --p_jump 0 --cam_interval 1 \
  --pitch_min_deg -30 --pitch_max_deg 30 \
  --run_name "alpha5" 2>&1 | tee -a logs/overnight/alpha5.log