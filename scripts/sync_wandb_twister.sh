#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${CONDA_ENV_NAME:=twister}"
: "${WANDB_PROJECT:=twister}"
: "${SYNC_TENSORBOARD:=1}"
: "${SYNC_WANDB_RUNS:=1}"
: "${SKIP_TENSORBOARD_WITH_WANDB_RUN:=1}"
: "${DRY_RUN:=0}"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

if [[ "${DRY_RUN}" != "1" ]] && ! wandb login --verify >/dev/null; then
  echo "[ERROR] wandb is not logged in for conda env '${CONDA_ENV_NAME}'."
  echo "        Run: conda activate ${CONDA_ENV_NAME} && wandb login"
  exit 1
fi

wandb_args=(--project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  wandb_args+=(--entity "${WANDB_ENTITY}")
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

wandb_run_roots=()

if [[ "${SYNC_WANDB_RUNS}" == "1" ]]; then
  mapfile -t wandb_run_dirs < <(
    find -L runs callbacks wandb -type f -name 'run-*.wandb' ! -path '*/latest-run/*' -printf '%h\n' 2>/dev/null | sort -u
  )
  for run_dir in "${wandb_run_dirs[@]}"; do
    root="${run_dir%/wandb/wandb/run-*}"
    if [[ "${root}" != "${run_dir}" ]]; then
      wandb_run_roots+=("${root}")
    fi
    echo "[INFO] Syncing W&B run: ${run_dir}"
    run_cmd wandb sync "${wandb_args[@]}" --include-online "${run_dir}"
  done
fi

if [[ "${SYNC_TENSORBOARD}" == "1" ]]; then
  mapfile -t tb_dirs < <(
    find -L runs callbacks -type f -name 'events.out.tfevents*' -printf '%h\n' 2>/dev/null | sort -u
  )
  for tb_dir in "${tb_dirs[@]}"; do
    if [[ "${SKIP_TENSORBOARD_WITH_WANDB_RUN}" == "1" ]]; then
      skip=0
      for root in "${wandb_run_roots[@]}"; do
        if [[ "${tb_dir}" == "${root}" || "${tb_dir}" == "${root}/"* ]]; then
          skip=1
          break
        fi
      done
      if [[ "${skip}" == "1" ]]; then
        echo "[INFO] Skipping TensorBoard logdir already covered by a W&B run: ${tb_dir}"
        continue
      fi
    fi
    echo "[INFO] Syncing TensorBoard logdir: ${tb_dir}"
    run_cmd wandb sync "${wandb_args[@]}" --sync-tensorboard "${tb_dir}"
  done
fi
