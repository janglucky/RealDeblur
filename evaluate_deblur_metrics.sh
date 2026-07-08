#!/usr/bin/env bash
set -e

: "${RESTORED_DIR:=/home/gd09385/work/RealDeblur/outputs/deblur_baseline_wrgb/checkpoint-50000}"
: "${GT_DIR:=/home/gd09385/data/test_c/target}"
: "${OUTPUT_DIR:=}"
: "${METRICS:=psnr ssim niqe lpips dists musiq maniqa clipiqa}"
: "${CROP_BORDER:=0}"
: "${TEST_Y_CHANNEL:=0}"
: "${NIQE_CONVERT_TO:=y}"
: "${RESIZE:=none}"
: "${DEVICE:=auto}"
: "${GT_MATCH:=auto}"
: "${MAX_IMAGES:=0}"
: "${SKIP_MISSING_GT:=0}"
: "${DISABLE_CUDNN:=1}"

if [ -z "${RESTORED_DIR}" ]; then
  echo "Please set RESTORED_DIR, for example: RESTORED_DIR=outputs/my_experiment/checkpoint-xxx"
  exit 1
fi

EXTRA_ARGS=()
if [ -n "${OUTPUT_DIR}" ]; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [ -n "${GT_DIR}" ]; then
  EXTRA_ARGS+=(--gt_dir "${GT_DIR}")
fi
if [ "${TEST_Y_CHANNEL}" = "1" ]; then
  EXTRA_ARGS+=(--test_y_channel)
fi
if [ "${MAX_IMAGES}" != "0" ]; then
  EXTRA_ARGS+=(--max_images "${MAX_IMAGES}")
fi
if [ "${SKIP_MISSING_GT}" = "1" ]; then
  EXTRA_ARGS+=(--skip_missing_gt)
fi
if [ "${DISABLE_CUDNN}" = "1" ]; then
  EXTRA_ARGS+=(--disable_cudnn)
fi

python evaluate_deblur_metrics.py \
  --restored_dir="${RESTORED_DIR}" \
  --metrics ${METRICS} \
  --crop_border="${CROP_BORDER}" \
  --niqe_convert_to="${NIQE_CONVERT_TO}" \
  --resize="${RESIZE}" \
  --device="${DEVICE}" \
  --gt_match="${GT_MATCH}" \
  "${EXTRA_ARGS[@]}"
