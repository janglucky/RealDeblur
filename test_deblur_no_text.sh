#!/usr/bin/env bash
set -e

: "${IMAGE_PATH:=/home/gd09385/data/test_c/source}"
: "${PRETRAINED_MODEL:=/home/gd09385/models/stable-diffusion-2-base}"
: "${PASD_MODEL:=experiments/pasd_deblur_no_text/checkpoint-10000}"
: "${OUTPUT_DIR:=outputs/pasd_deblur_no_text}"
: "${MIXED_PRECISION:=fp16}"
: "${NUM_INFERENCE_STEPS:=20}"
: "${CONDITIONING_SCALE:=1.0}"
: "${PROCESS_SIZE:=0}"
: "${DECODER_TILED_SIZE:=224}"
: "${ENCODER_TILED_SIZE:=1024}"
: "${LATENT_TILED_SIZE:=320}"
: "${LATENT_TILED_OVERLAP:=8}"
: "${DISABLE_CUDNN:=1}"
: "${DISABLE_XFORMERS:=0}"
: "${MAX_IMAGES:=0}"
: "${MIN_FREE_VRAM_MB:=8000}"
: "${SHOW_PROGRESS:=0}"
: "${SEED:=}"

if [ ! -d "${PASD_MODEL}/unet" ] || [ ! -d "${PASD_MODEL}/controlnet" ]; then
  echo "Missing trained model under ${PASD_MODEL}."
  echo "Expected ${PASD_MODEL}/unet and ${PASD_MODEL}/controlnet."
  exit 1
fi

EXTRA_ARGS=()
if [ "${DISABLE_CUDNN}" = "1" ]; then
  EXTRA_ARGS+=(--disable_cudnn)
fi
if [ "${DISABLE_XFORMERS}" = "1" ]; then
  EXTRA_ARGS+=(--disable_xformers)
fi
if [ "${MAX_IMAGES}" != "0" ]; then
  EXTRA_ARGS+=(--max_images "${MAX_IMAGES}")
fi
if [ "${SHOW_PROGRESS}" = "1" ]; then
  EXTRA_ARGS+=(--show_progress)
fi
if [ -n "${SEED}" ]; then
  EXTRA_ARGS+=(--seed "${SEED}")
fi

python test_deblur_no_text.py \
  --pretrained_model_path="${PRETRAINED_MODEL}" \
  --pasd_model_path="${PASD_MODEL}" \
  --image_path="${IMAGE_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --mixed_precision="${MIXED_PRECISION}" \
  --num_inference_steps="${NUM_INFERENCE_STEPS}" \
  --conditioning_scale="${CONDITIONING_SCALE}" \
  --process_size="${PROCESS_SIZE}" \
  --decoder_tiled_size="${DECODER_TILED_SIZE}" \
  --encoder_tiled_size="${ENCODER_TILED_SIZE}" \
  --latent_tiled_size="${LATENT_TILED_SIZE}" \
  --latent_tiled_overlap="${LATENT_TILED_OVERLAP}" \
  --min_free_vram_mb="${MIN_FREE_VRAM_MB}" \
  "${EXTRA_ARGS[@]}"
