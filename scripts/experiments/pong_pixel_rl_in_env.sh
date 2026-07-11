#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [real|wm] [wm_checkpoint] [run_name]"
  echo "Examples:"
  echo "  $0 real '' pong_pixel_rl_in_env/logdir/real"
  echo "  $0 wm callbacks/atari100k/atari100k-pong/checkpoints_100000.ckpt pong_pixel_rl_in_env/logdir/wm_twister"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${CONDA_ENV_NAME:=twister}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PIXEL_RL_AC_UPDATES:=20000}"
: "${PIXEL_RL_STEPS:=100000}"
: "${PIXEL_RL_ENVS:=64}"
: "${PIXEL_RL_BACKUP_EVERY:=15}"
: "${PIXEL_RL_LOG_EVERY:=1000}"
: "${PIXEL_RL_SAVE_EVERY:=10000}"
: "${PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY:=10000}"
: "${PIXEL_RL_WM_HORIZON:=512}"
: "${PIXEL_RL_WM_RESPECT_TERMINAL:=True}"
: "${PIXEL_RL_WM_INITIAL_SOURCE:=real}"
: "${PIXEL_RL_WM_BOOTSTRAP_DATASET:=}"
: "${PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD:=0.5}"
: "${PIXEL_RL_EVAL_REAL_EVERY:=2000}"
: "${PIXEL_RL_EVAL_REAL_VIDEO_EVERY:=10000}"
: "${PIXEL_RL_EVAL_REAL_EPS:=5}"
: "${PIXEL_RL_RESUME:=True}"
: "${WANDB_ENABLED:=0}"
: "${WANDB_PROJECT:=twister}"
: "${WANDB_MODE:=online}"
export CUDA_VISIBLE_DEVICES WANDB_ENABLED WANDB_PROJECT WANDB_MODE

BACKEND="${1:-real}"
WM_CHECKPOINT="${2:-}"
RUN_NAME="${3:-pong_pixel_rl_in_env/logdir/${BACKEND}}"

if [[ -n "${PIXEL_RL_AC_UPDATES}" ]]; then
  PIXEL_RL_STEPS=$(( PIXEL_RL_AC_UPDATES * PIXEL_RL_BACKUP_EVERY * PIXEL_RL_ENVS ))
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "[INFO] Backend: ${BACKEND}"
echo "[INFO] Run name: ${RUN_NAME}"
echo "[INFO] WM checkpoint: ${WM_CHECKPOINT}"
echo "[INFO] Steps: ${PIXEL_RL_STEPS}"
echo "[INFO] AC updates: ${PIXEL_RL_AC_UPDATES:-<derived from steps>}"
echo "[INFO] Parallel envs: ${PIXEL_RL_ENVS}"
echo "[INFO] WM horizon: ${PIXEL_RL_WM_HORIZON}"
echo "[INFO] WM initial source: ${PIXEL_RL_WM_INITIAL_SOURCE}"
echo "[INFO] WM bootstrap dataset: ${PIXEL_RL_WM_BOOTSTRAP_DATASET}"
echo "[INFO] W&B project: ${WANDB_PROJECT}"

python -u -m pixel_rl.train \
  --backend "${BACKEND}" \
  --run-name "${RUN_NAME}" \
  --env-name "PongNoFrameskip-v4" \
  --seed "${SEED:-42}" \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --steps "${PIXEL_RL_STEPS}" \
  --ac-updates "${PIXEL_RL_AC_UPDATES}" \
  --envs "${PIXEL_RL_ENVS}" \
  --device "${PIXEL_RL_DEVICE:-cuda}" \
  --wm-horizon "${PIXEL_RL_WM_HORIZON}" \
  --wm-respect-terminal "${PIXEL_RL_WM_RESPECT_TERMINAL}" \
  --wm-initial-source "${PIXEL_RL_WM_INITIAL_SOURCE}" \
  --wm-bootstrap-dataset "${PIXEL_RL_WM_BOOTSTRAP_DATASET}" \
  --wm-reward-quantize-threshold "${PIXEL_RL_WM_REWARD_QUANTIZE_THRESHOLD}" \
  --backup-every "${PIXEL_RL_BACKUP_EVERY}" \
  --log-every "${PIXEL_RL_LOG_EVERY}" \
  --save-every "${PIXEL_RL_SAVE_EVERY}" \
  --wm-rollout-video-every "${PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY}" \
  --eval-real-every "${PIXEL_RL_EVAL_REAL_EVERY}" \
  --eval-real-video-every "${PIXEL_RL_EVAL_REAL_VIDEO_EVERY}" \
  --eval-real-eps "${PIXEL_RL_EVAL_REAL_EPS}" \
  --resume "${PIXEL_RL_RESUME}" \
  --lr "${PIXEL_RL_LR:-1e-4}" \
  --eps "${PIXEL_RL_EPS:-1e-8}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE}" \
  --wandb-enabled "${WANDB_ENABLED}"
