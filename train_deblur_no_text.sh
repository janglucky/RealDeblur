#!/usr/bin/env bash
set -e

: "${TRAIN_BLUR_DIR:=/home/gd09385/data/test_c_sub/source}"
: "${TRAIN_SHARP_DIR:=/home/gd09385/data/test_c_sub/target}"
: "${PRETRAINED_MODEL:=/home/gd09385/models/stable-diffusion-2-base}"
: "${OUTPUT_DIR:=experiments/pasd_deblur_no_text}"
: "${RESOLUTION:=512}"
: "${LEARNING_RATE:=5e-5}"
: "${GRADIENT_ACCUMULATION_STEPS:=4}"
: "${TRAIN_BATCH_SIZE:=1}"
: "${NUM_TRAIN_EPOCHS:=1000}"
: "${CHECKPOINTING_STEPS:=5000}"
: "${MIXED_PRECISION:=fp16}"
: "${DATALOADER_NUM_WORKERS:=8}"
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

accelerate launch train_deblur_no_text.py \
  --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
  --train_blur_dir="${TRAIN_BLUR_DIR}" \
  --train_sharp_dir="${TRAIN_SHARP_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --resolution="${RESOLUTION}" \
  --learning_rate="${LEARNING_RATE}" \
  --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --train_batch_size="${TRAIN_BATCH_SIZE}" \
  --num_train_epochs="${NUM_TRAIN_EPOCHS}" \
  --tracker_project_name="pasd_deblur_no_text" \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps="${CHECKPOINTING_STEPS}" \
  --mixed_precision="${MIXED_PRECISION}" \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"
