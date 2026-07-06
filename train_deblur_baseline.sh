#!/usr/bin/env bash
set -e

: "${TRAIN_BLUR_DIR:=/home/gd09385/data/test_c_sub/source}"
: "${TRAIN_SHARP_DIR:=/home/gd09385/data/test_c_sub/target}"
: "${PRETRAINED_MODEL:=/home/gd09385/models/stable-diffusion-2-base}"
: "${OUTPUT_DIR:=experiments/deblur_baseline}"
: "${RESOLUTION:=512}"
: "${LEARNING_RATE:=5e-5}"
: "${GRADIENT_ACCUMULATION_STEPS:=4}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${MAX_TRAIN_STEPS:=20000}"
: "${CHECKPOINTING_STEPS:=5000}"
: "${MIXED_PRECISION:=fp16}"
: "${DATALOADER_NUM_WORKERS:=8}"
: "${REPORT_TO:=none}"
: "${DISABLE_CUDNN:=1}"
: "${GRADIENT_CHECKPOINTING:=1}"
: "${USE_8BIT_ADAM:=0}"

EXTRA_ARGS=()
if [ "${DISABLE_CUDNN}" = "1" ]; then
  EXTRA_ARGS+=(--disable_cudnn)
fi
if [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
  EXTRA_ARGS+=(--gradient_checkpointing)
fi
if [ "${USE_8BIT_ADAM}" = "1" ]; then
  EXTRA_ARGS+=(--use_8bit_adam)
fi

accelerate launch train_deblur_baseline.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
  --train_blur_dir="${TRAIN_BLUR_DIR}" \
  --train_sharp_dir="${TRAIN_SHARP_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --resolution="${RESOLUTION}" \
  --learning_rate="${LEARNING_RATE}" \
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --train_batch_size="${TRAIN_BATCH_SIZE}" \
  --max_train_steps="${MAX_TRAIN_STEPS}" \
  --tracker_project_name="deblur_baseline" \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps="${CHECKPOINTING_STEPS}" \
  --mixed_precision="${MIXED_PRECISION}" \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  --report_to="${REPORT_TO}" \
  "${EXTRA_ARGS[@]}"
