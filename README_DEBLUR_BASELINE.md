# PASD Deblur Baseline Without Text Branch

This baseline removes the runtime text branch from PASD:

- no caption files are read;
- no tokenizer or CLIP text encoder is loaded;
- prompt generation via classification, detection, BLIP, or CoCa is not used;
- UNet and ControlNet still receive an `encoder_hidden_states` tensor, but it is a fixed zero condition with shape `(B, 77, cross_attention_dim)` so existing SD/PASD weights remain compatible.

## Data Layout

Use paired blur/sharp folders. Relative paths are matched first; if that fails, filenames with the same stem are matched.

```text
datasets/deblur/train/blur/0001.png
datasets/deblur/train/blur/0002.png
datasets/deblur/train/sharp/0001.png
datasets/deblur/train/sharp/0002.png
```

## Train

```bash
TRAIN_BLUR_DIR=/path/to/train/blur \
TRAIN_SHARP_DIR=/path/to/train/sharp \
PRETRAINED_MODEL=checkpoints/stable-diffusion-v1-5 \
OUTPUT_DIR=runs/pasd_deblur_no_text \
bash train_deblur_no_text.sh
```

The trained model is saved as:

```text
runs/pasd_deblur_no_text/unet/
runs/pasd_deblur_no_text/controlnet/
```

## Test

```bash
python test_deblur_no_text.py \
  --pretrained_model_path checkpoints/stable-diffusion-v1-5 \
  --pasd_model_path runs/pasd_deblur_no_text \
  --image_path /path/to/blur_or_folder \
  --output_dir output_deblur_no_text \
  --guidance_scale 1.0
```

`--guidance_scale` defaults to `1.0` because classifier-free text guidance has been removed.
