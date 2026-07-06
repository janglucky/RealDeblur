import argparse
import glob
import os
import subprocess

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image

from pasd.pipelines.pipeline_pasd import StableDiffusionControlNetPipeline


check_min_version("0.18.0.dev0")
logger = get_logger(__name__, log_level="INFO")
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def get_model_classes():
    from pasd.models.pasd.controlnet import ControlNetModel
    from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel

    return UNet2DConditionModel, ControlNetModel


def resize_for_pipeline(image, process_size):
    original_size = image.size
    width, height = image.size

    if process_size > 0 and min(width, height) < process_size:
        scale = process_size / min(width, height)
        width = round(width * scale)
        height = round(height * scale)

    width = max(width // 8 * 8, 8)
    height = max(height // 8 * 8, 8)
    if (width, height) != image.size:
        image = image.resize((width, height), Image.BICUBIC)
    return image, original_size


def load_pasd_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    UNet2DConditionModel, ControlNetModel = get_model_classes()

    scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pasd_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.pasd_model_path, subfolder="controlnet")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    pipeline._init_tiled_vae(
        encoder_tile_size=args.encoder_tiled_size,
        decoder_tile_size=args.decoder_tiled_size,
    )
    return pipeline


def main(args, enable_xformers_memory_efficient_attention=True):
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN is disabled for this run.")

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if torch.cuda.is_available() and args.min_free_vram_mb > 0:
        free_bytes, total_bytes = torch.cuda.mem_get_info(accelerator.device)
        free_mb = free_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
        if free_mb < args.min_free_vram_mb:
            print(
                f"Not enough free VRAM for testing: {free_mb} MiB free / {total_mb} MiB total, "
                f"need at least {args.min_free_vram_mb} MiB."
            )
            try:
                apps = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                if apps:
                    print("GPU compute processes:")
                    print(apps)
            except Exception:
                pass
            raise SystemExit(1)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    pipeline = load_pasd_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=not args.show_progress)

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    if os.path.isdir(args.image_path):
        image_names = sorted(
            path for path in glob.glob(os.path.join(args.image_path, "*")) if path.lower().endswith(IMG_EXTENSIONS)
        )
    else:
        image_names = [args.image_path]
    if args.max_images > 0:
        image_names = image_names[: args.max_images]

    for image_name in image_names:
        print(f"Processing {image_name}")
        blur_image = Image.open(image_name).convert("RGB")
        condition_image, original_size = resize_for_pipeline(blur_image, args.process_size)

        try:
            with torch.autocast(accelerator.device.type, enabled=accelerator.device.type == "cuda"):
                image = pipeline(
                    args,
                    image=condition_image,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    conditioning_scale=args.conditioning_scale,
                ).images[0]
        except Exception as exc:
            logger.error(f"Failed on {image_name}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if image.size != original_size:
            image = image.resize(original_size, Image.BICUBIC)

        name, _ = os.path.splitext(os.path.basename(image_name))
        image.save(os.path.join(args.output_dir, f"{name}.png"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="checkpoints/stable-diffusion-v1-5")
    parser.add_argument("--pasd_model_path", type=str, default="runs/deblur_baseline")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_deblur_baseline")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--process_size", type=int, default=0)
    parser.add_argument("--decoder_tiled_size", type=int, default=224)
    parser.add_argument("--encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=320)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--init_latent_with_noise", action="store_true")
    parser.add_argument("--added_noise_level", type=int, default=900)
    parser.add_argument("--offset_noise_scale", type=float, default=0.0)
    parser.add_argument("--disable_cudnn", action="store_true")
    parser.add_argument("--disable_xformers", action="store_true")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--min_free_vram_mb", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show_progress", action="store_true")
    args = parser.parse_args()
    main(args, enable_xformers_memory_efficient_attention=not args.disable_xformers)
