# RealDeblur

RealDeblur is a paired image deblurring baseline derived from PASD. This repository keeps the Stable Diffusion + ControlNet restoration core and removes components that are not needed for the deblurring experiment.

## Scope

This codebase is focused on one task:

- train a PASD-style deblurring baseline on paired blurry/sharp images
- run inference on blurry images with the trained UNet and ControlNet weights

Removed from this baseline:

- tokenizer, CLIP text encoder, prompts, captions, textual inversion, and classifier-free text guidance
- super-resolution, colorization, stylization, SDXL, PASD-light, personalized models, Gradio, annotators, RealESRGAN, webdataset, and placeholder checkpoints
- auxiliary RGB restoration heads and their extra reconstruction losses

The UNet and ControlNet still need cross-attention tensors for shape compatibility with Stable Diffusion 2-base. The text branch is replaced by a fixed zero-valued cross-attention condition.

## Installation

Create an environment with PyTorch matching your CUDA driver, then install the repo dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The default scripts use Stable Diffusion 2-base from:

```text
/home/gd09385/models/stable-diffusion-2-base
```

That directory should contain the usual Diffusers layout:

```text
stable-diffusion-2-base/
  scheduler/
  unet/
  vae/
  model_index.json
```

Override the path with `PRETRAINED_MODEL=/path/to/stable-diffusion-2-base` when needed.

## Data Layout

Use paired blurry and sharp images. Pairs are matched first by relative path, then by filename stem.

```text
data/deblur/
  train/
    blur/
      0001.png
      0002.png
    sharp/
      0001.png
      0002.png
  test/
    blur/
      0003.png
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`.

## Training

The included script is configured for the current machine paths:

```bash
bash train_deblur_baseline.sh
```

Common overrides:

```bash
PRETRAINED_MODEL=/home/gd09385/models/stable-diffusion-2-base \
TRAIN_BLUR_DIR=/home/gd09385/data/test_c_sub/source \
TRAIN_SHARP_DIR=/home/gd09385/data/test_c_sub/target \
OUTPUT_DIR=experiments/deblur_baseline \
TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
MAX_TRAIN_STEPS=20000 \
bash train_deblur_baseline.sh
```

Useful options:

- `DISABLE_CUDNN=1` disables cuDNN. Keep this on if the machine reports `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
- `GRADIENT_CHECKPOINTING=1` reduces VRAM usage.
- `USE_8BIT_ADAM=1` enables bitsandbytes AdamW if bitsandbytes is installed.
- `MAX_TRAIN_STEPS=20000` sets the total number of optimizer steps. The script derives the required epoch count from this value.
- `CHECKPOINTING_STEPS=5000` controls checkpoint frequency.
- `REPORT_TO=tensorboard` enables TensorBoard logging if TensorBoard is installed. The default is `none`.

Checkpoints are written under `experiments/` by default and are ignored by git.

## Testing

Run inference after a checkpoint has `unet/` and `controlnet/` subdirectories:

```bash
bash test_deblur_baseline.sh
```

By default, the script searches `CHECKPOINT_DIR=experiments/deblur_baseline` and uses the numerically latest `checkpoint-*` directory.
If `OUTPUT_DIR` is not set, outputs are written to `OUTPUT_ROOT/<experiment-name>/<checkpoint-name>`, for example `outputs/deblur_baseline/checkpoint-20000`.

Use a specific training step:

```bash
CHECKPOINT_DIR=experiments/deblur_baseline \
CHECKPOINT_STEP=20000 \
bash test_deblur_baseline.sh
```

Use an explicit checkpoint path:

```bash
PRETRAINED_MODEL=/home/gd09385/models/stable-diffusion-2-base \
PASD_MODEL=experiments/deblur_baseline/checkpoint-10000 \
IMAGE_PATH=/home/gd09385/data/test_c/source \
MAX_IMAGES=5 \
bash test_deblur_baseline.sh
```

Set `OUTPUT_ROOT=/path/to/results` to change the root directory while keeping the experiment/checkpoint subfolders.

Testing has a startup free-VRAM check. The default is `MIN_FREE_VRAM_MB=8000`; lower it only if you know the checkpoint and image size fit your GPU.

## GPU Notes

- Do not run training and testing at the same time on the same GPU unless there is enough free VRAM.
- If a killed process leaves VRAM occupied, check `nvidia-smi` for remaining Python processes and terminate the owning PID.
- Outputs, checkpoints, caches, and local experiment artifacts are intentionally ignored by git.

## Main Files

- `train_deblur_baseline.py`: paired deblurring training
- `test_deblur_baseline.py`: deblurring baseline inference
- `pasd/dataloader/deblur.py`: paired blur/sharp dataset
- `pasd/pipelines/pipeline_pasd.py`: text-branch-removed SD + ControlNet pipeline
- `pasd/myutils/null_condition.py`: zero cross-attention condition helper
