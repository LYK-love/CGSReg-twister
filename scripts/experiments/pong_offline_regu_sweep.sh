#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${CONDA_ENV_NAME:=twister}"
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${DATASET_PATH:=/SSD_RAID0/lyk/shared_replay/diamond_pong_for_simulus_twister/twister/train}"
: "${EVAL_DATASET_PATH:=}"
: "${EVAL_EVERY:=0}"
: "${EVAL_BATCHES:=8}"
: "${EVAL_BATCH_SIZE:=0}"
: "${MODEL_SIZES:=base}"
: "${WM_INITIAL_SOURCE:=real}"
: "${SPATIAL_WEIGHTS:=0.01,0.1,1.0,10.0}"
: "${MASK_PRESETS:=mask1}"
: "${TRAIN_STEPS:=100000}"
: "${BATCH_SIZE:=16}"
: "${BATCH_LENGTH:=64}"
: "${TORCH_NUM_THREADS:=8}"
: "${LOG_EVERY:=10}"
: "${COMMAND_FILE:=scripts/experiments/pong_offline_regu_base_mask1.commands.txt}"
: "${PROJECT_ROOT:=$(pwd)}"
: "${RUN_SWEEP:=0}"
: "${SCHEDULER_CUDA_DEVICES:=auto}"
: "${SCHEDULER_LOG_DIR:=runs/pong_offline_regu_sweep/scheduler_logs}"
: "${SCHEDULER_EXTRA_ARGS:=--verbose --keep-job-tabs}"
: "${WANDB_ENABLED:=1}"
: "${WANDB_PROJECT:=twister}"
: "${WANDB_ENTITY:=ssl-lab}"
: "${WANDB_MODE:=online}"

export CUDA_VISIBLE_DEVICES WANDB_ENABLED WANDB_PROJECT WANDB_ENTITY WANDB_MODE

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python scripts/experiments/generate_pong_offline_regu_commands.py \
  --dataset-path "${DATASET_PATH}" \
  --eval-dataset-path "${EVAL_DATASET_PATH}" \
  --eval-every "${EVAL_EVERY}" \
  --eval-batches "${EVAL_BATCHES}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --model-sizes "${MODEL_SIZES}" \
  --wm-initial-source "${WM_INITIAL_SOURCE}" \
  --spatial-weights "${SPATIAL_WEIGHTS}" \
  --mask-presets "${MASK_PRESETS}" \
  --train-steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --batch-length "${BATCH_LENGTH}" \
  --torch-num-threads "${TORCH_NUM_THREADS}" \
  --log-every "${LOG_EVERY}" \
  --wandb-enabled "${WANDB_ENABLED}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-mode "${WANDB_MODE}" \
  --project-root "${PROJECT_ROOT}" \
  --scheduler-logs-dir "${SCHEDULER_LOG_DIR}" \
  --output "${COMMAND_FILE}"

echo "[INFO] Generated ${COMMAND_FILE}"
echo "[INFO] Run with tiny-exp-scheduler:"
echo "tiny-exp-scheduler run ${COMMAND_FILE} --cuda-devices ${SCHEDULER_CUDA_DEVICES} --logs-dir ${SCHEDULER_LOG_DIR} ${SCHEDULER_EXTRA_ARGS}"

if [[ "${RUN_SWEEP}" == "1" ]]; then
  # shellcheck disable=SC2086
  tiny-exp-scheduler run "${COMMAND_FILE}" \
    --cuda-devices "${SCHEDULER_CUDA_DEVICES}" \
    --logs-dir "${SCHEDULER_LOG_DIR}" \
    ${SCHEDULER_EXTRA_ARGS}
fi
