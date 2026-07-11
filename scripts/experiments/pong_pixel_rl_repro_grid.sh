#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${THRESHOLDS:=0.01,0.1,0.5}"
: "${HORIZONS:=128,512}"
: "${AC_UPDATES:=20000}"
: "${PIXEL_RL_ENVS:=64}"
: "${PIXEL_RL_BACKUP_EVERY:=15}"
: "${PIXEL_RL_LOG_EVERY:=1000}"
: "${PIXEL_RL_SAVE_EVERY:=10000}"
: "${PIXEL_RL_EVAL_REAL_EVERY:=2000}"
: "${PIXEL_RL_EVAL_REAL_VIDEO_EVERY:=10000}"
: "${PIXEL_RL_EVAL_REAL_EPS:=5}"
: "${PIXEL_RL_RESUME:=False}"
: "${PIXEL_RL_WM_INITIAL_SOURCE:=real}"
: "${PIXEL_RL_WM_RESPECT_TERMINAL:=True}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${WANDB_ENABLED:=1}"
: "${WANDB_ENTITY:=ssl-lab}"
: "${WANDB_MODE:=online}"
: "${HOST_TAG:=scorpio}"

python scripts/experiments/generate_pong_pixel_rl_repro_grid_commands.py \
  --thresholds "${THRESHOLDS}" \
  --horizons "${HORIZONS}" \
  --ac-updates "${AC_UPDATES}" \
  --envs "${PIXEL_RL_ENVS}" \
  --backup-every "${PIXEL_RL_BACKUP_EVERY}" \
  --log-every "${PIXEL_RL_LOG_EVERY}" \
  --save-every "${PIXEL_RL_SAVE_EVERY}" \
  --eval-real-every "${PIXEL_RL_EVAL_REAL_EVERY}" \
  --eval-real-video-every "${PIXEL_RL_EVAL_REAL_VIDEO_EVERY}" \
  --eval-real-eps "${PIXEL_RL_EVAL_REAL_EPS}" \
  --resume "${PIXEL_RL_RESUME}" \
  --wm-initial-source "${PIXEL_RL_WM_INITIAL_SOURCE}" \
  --wm-respect-terminal "${PIXEL_RL_WM_RESPECT_TERMINAL}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --wandb-enabled "${WANDB_ENABLED}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-mode "${WANDB_MODE}" \
  --host-tag "${HOST_TAG}"

echo "[INFO] Generated TWISTER commands:"
echo "  scripts/experiments/pong_pixel_rl_twister_repro_grid.commands.txt"
echo "[INFO] Generated STORM commands:"
echo "  /scorpio/home/luyukuan/projects/oc-storm/scripts/experiments/pong_pixel_rl_storm_repro_grid.commands.txt"
echo "[INFO] No jobs were launched."
