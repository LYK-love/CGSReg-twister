#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${CONDA_ENV_NAME:=twister}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${WM_CHECKPOINT:=callbacks/atari100k/atari100k-pong/checkpoints_epoch_50_step_100000.ckpt}"
: "${THRESHOLDS:=0.1,0.25,0.5,0.75,1.0}"
: "${AC_UPDATES:=20000}"
: "${PIXEL_RL_ENVS:=64}"
: "${PIXEL_RL_BACKUP_EVERY:=15}"
: "${PIXEL_RL_WM_HORIZON:=512}"
: "${PIXEL_RL_WM_INITIAL_SOURCE:=real}"
: "${PIXEL_RL_WM_BOOTSTRAP_DATASET:=datasets/converted_from_diamond/pong/train}"
: "${PIXEL_RL_WM_RESPECT_TERMINAL:=True}"
: "${PIXEL_RL_EVAL_REAL_EVERY:=2000}"
: "${PIXEL_RL_EVAL_REAL_VIDEO_EVERY:=10000}"
: "${PIXEL_RL_EVAL_REAL_EPS:=5}"
: "${PIXEL_RL_LOG_EVERY:=1000}"
: "${PIXEL_RL_SAVE_EVERY:=10000}"
: "${PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY:=10000}"
: "${PIXEL_RL_RESUME:=False}"
: "${WANDB_ENABLED:=1}"
: "${WANDB_PROJECT:=rl-in-pixel-env}"
: "${WANDB_ENTITY:=ssl-lab}"
: "${WANDB_MODE:=online}"
: "${HOST_TAG:=scorpio}"
: "${COMMAND_FILE:=scripts/experiments/pong_pixel_rl_rewq_sweep.commands.txt}"
: "${SCHEDULER_LOG_DIR:=pong_pixel_rl_in_env/scheduler_logs/rewq_sweep}"
: "${RUN_SWEEP:=0}"
: "${SCHEDULER_CUDA_DEVICES:=auto}"
: "${SCHEDULER_EXTRA_ARGS:=--verbose --keep-job-tabs}"

export CUDA_VISIBLE_DEVICES CONDA_ENV_NAME WANDB_ENABLED WANDB_PROJECT WANDB_ENTITY WANDB_MODE

python scripts/experiments/generate_pong_pixel_rl_rewq_sweep_commands.py \
  --wm-checkpoint "${WM_CHECKPOINT}" \
  --thresholds "${THRESHOLDS}" \
  --ac-updates "${AC_UPDATES}" \
  --envs "${PIXEL_RL_ENVS}" \
  --backup-every "${PIXEL_RL_BACKUP_EVERY}" \
  --wm-horizon "${PIXEL_RL_WM_HORIZON}" \
  --wm-initial-source "${PIXEL_RL_WM_INITIAL_SOURCE}" \
  --wm-bootstrap-dataset "${PIXEL_RL_WM_BOOTSTRAP_DATASET}" \
  --respect-terminal "${PIXEL_RL_WM_RESPECT_TERMINAL}" \
  --eval-real-every "${PIXEL_RL_EVAL_REAL_EVERY}" \
  --eval-real-video-every "${PIXEL_RL_EVAL_REAL_VIDEO_EVERY}" \
  --eval-real-eps "${PIXEL_RL_EVAL_REAL_EPS}" \
  --log-every "${PIXEL_RL_LOG_EVERY}" \
  --save-every "${PIXEL_RL_SAVE_EVERY}" \
  --wm-rollout-video-every "${PIXEL_RL_WM_ROLLOUT_VIDEO_EVERY}" \
  --resume "${PIXEL_RL_RESUME}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --conda-env-name "${CONDA_ENV_NAME}" \
  --wandb-enabled "${WANDB_ENABLED}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-mode "${WANDB_MODE}" \
  --host-tag "${HOST_TAG}" \
  --output "${COMMAND_FILE}" \
  --scheduler-logs-dir "${SCHEDULER_LOG_DIR}"

echo "[INFO] Generated ${COMMAND_FILE}"
echo "[INFO] Only reward threshold is swept. Terminal stays boolean and is controlled by PIXEL_RL_WM_RESPECT_TERMINAL=${PIXEL_RL_WM_RESPECT_TERMINAL}."
echo "[INFO] Run with tiny-exp-scheduler:"
echo "tiny-exp-scheduler run ${COMMAND_FILE} --cuda-devices ${SCHEDULER_CUDA_DEVICES} --logs-dir ${SCHEDULER_LOG_DIR} ${SCHEDULER_EXTRA_ARGS}"

if [[ "${RUN_SWEEP}" == "1" ]]; then
  # shellcheck disable=SC2086
  tiny-exp-scheduler run "${COMMAND_FILE}" \
    --cuda-devices "${SCHEDULER_CUDA_DEVICES}" \
    --logs-dir "${SCHEDULER_LOG_DIR}" \
    ${SCHEDULER_EXTRA_ARGS}
fi
