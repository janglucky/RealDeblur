#!/usr/bin/env bash
set -e

: "${IMAGE_PATH:=/home/gd09385/data/test_c/source}"
: "${PRETRAINED_MODEL:=/home/gd09385/models/stable-diffusion-2-base}"
: "${CHECKPOINT_DIR:=experiments/deblur_baseline_wrgb_sa}"
: "${CHECKPOINT_STEP:=}"
: "${PASD_MODEL:=}"
: "${OUTPUT_ROOT:=outputs}"
: "${OUTPUT_DIR:=}"
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
: "${COLOR_FIX_TYPE:=wavelet}"
: "${USE_NULL_PROMPT:=0}"

if [ -z "${PASD_MODEL}" ]; then
  if [ -n "${CHECKPOINT_STEP}" ]; then
    STEP="${CHECKPOINT_STEP#checkpoint-}"
    if [[ ! "${STEP}" =~ ^[0-9]+$ ]]; then
      echo "Invalid CHECKPOINT_STEP=${CHECKPOINT_STEP}. Use a number like 20000."
      exit 1
    fi
    PASD_MODEL="${CHECKPOINT_DIR}/checkpoint-${STEP}"
  else
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
      echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}"
      exit 1
    fi

    LATEST_STEP=-1
    LATEST_CHECKPOINT=""
    shopt -s nullglob
    for CHECKPOINT_PATH in "${CHECKPOINT_DIR}"/checkpoint-*; do
      [ -d "${CHECKPOINT_PATH}" ] || continue
      CHECKPOINT_NAME="$(basename "${CHECKPOINT_PATH}")"
      STEP="${CHECKPOINT_NAME#checkpoint-}"
      if [[ "${STEP}" =~ ^[0-9]+$ ]] && [ "${STEP}" -gt "${LATEST_STEP}" ]; then
        LATEST_STEP="${STEP}"
        LATEST_CHECKPOINT="${CHECKPOINT_PATH}"
      fi
    done
    shopt -u nullglob

    if [ -z "${LATEST_CHECKPOINT}" ]; then
      echo "No checkpoint-* directories found under ${CHECKPOINT_DIR}."
      exit 1
    fi
    PASD_MODEL="${LATEST_CHECKPOINT}"
  fi
fi

if [ -z "${OUTPUT_DIR}" ]; then
  CHECKPOINT_NAME="$(basename "${PASD_MODEL}")"
  EXPERIMENT_NAME="$(basename "$(dirname "${PASD_MODEL}")")"
  OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}/${CHECKPOINT_NAME}"
fi

if [ ! -d "${PASD_MODEL}/unet" ] || [ ! -d "${PASD_MODEL}/controlnet" ]; then
  echo "Missing trained model under ${PASD_MODEL}."
  echo "Expected ${PASD_MODEL}/unet and ${PASD_MODEL}/controlnet."
  exit 1
fi

echo "Using checkpoint: ${PASD_MODEL}"
echo "Writing outputs to: ${OUTPUT_DIR}"
echo "Color fix: ${COLOR_FIX_TYPE}"
if [ "${USE_NULL_PROMPT}" = "1" ]; then
  echo "Attention condition: zero null prompt"
else
  echo "Attention condition: self-conditioned hidden states"
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
if [ "${USE_NULL_PROMPT}" = "1" ]; then
  EXTRA_ARGS+=(--use_null_prompt)
fi

python test_deblur_baseline.py \
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
  --color_fix_type="${COLOR_FIX_TYPE}" \
  "${EXTRA_ARGS[@]}"
