#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

: "${PYTHON:=python}"
: "${CONVERTER:=/scorpio/home/luyukuan/projects/oc-storm/scripts/dataset/convert_diamond_replay.py}"
: "${DIAMOND_ROOT:=/data/luyukuan/projects/diamond-assets/datasets/pong}"
: "${OUTPUT_ROOT:=/data/luyukuan/projects/oc-storm/datasets/converted_from_diamond/pong}"
: "${IMAGE_SIZE:=64}"
: "${INCLUDE_MASKS:=1}"
: "${FORCE:=0}"

args=(
  "${CONVERTER}"
  --diamond-root "${DIAMOND_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --image-size "${IMAGE_SIZE}"
  --no-ram
)

if [[ "${INCLUDE_MASKS}" == "1" ]]; then
  args+=(--include-masks)
fi

if [[ "${FORCE}" == "1" ]]; then
  args+=(--force)
fi

"${PYTHON}" "${args[@]}"

echo "[INFO] Train split: ${OUTPUT_ROOT}/train"
echo "[INFO] Eval split: ${OUTPUT_ROOT}/eval"
