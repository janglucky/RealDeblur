#!/usr/bin/env python
import argparse
import logging
import math
import os
from pathlib import Path

import accelerate
import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pasd.dataloader.deblur import DeblurPairDataset
from pasd.myutils.null_condition import build_null_encoder_hidden_states, get_cross_attention_dim
from pasd.pipelines.pipeline_pasd import StableDiffusionControlNetPipeline


check_min_version("0.15.0.dev0")
logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="PASD deblurring baseline without the text branch.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--train_blur_dir", type=str, required=True)
    parser.add_argument("--train_sharp_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/deblur_baseline")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--no_random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default="deblur_baseline")
    parser.add_argument(
        "--trainable_modules",
        nargs="*",
        type=str,
        default=["pixel_attentions", "norm2_plus", "attn2_plus", "proj_in_plus"],
    )
    parser.add_argument("--validation_blur", type=str, default=None, nargs="*")
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--num_validation_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--decoder_tiled_size", type=int, default=224)
    parser.add_argument("--encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=320)
    parser.add_argument("--latent_tiled_overlap", type=int, default=8)
    parser.add_argument("--init_latent_with_noise", action="store_true")
    parser.add_argument("--added_noise_level", type=int, default=900)
    parser.add_argument("--offset_noise_scale", type=float, default=0.0)
    parser.add_argument("--disable_cudnn", action="store_true")

    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()

    if not os.path.isdir(args.train_blur_dir):
        raise ValueError(f"`--train_blur_dir` does not exist: {args.train_blur_dir}")
    if not os.path.isdir(args.train_sharp_dir):
        raise ValueError(f"`--train_sharp_dir` does not exist: {args.train_sharp_dir}")
    return args


def get_model_classes():
    from pasd.models.pasd.controlnet import ControlNetModel
    from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel

    return UNet2DConditionModel, ControlNetModel


def load_controlnet(ControlNetModel, model_path):
    if os.path.isdir(os.path.join(model_path, "controlnet")):
        return ControlNetModel.from_pretrained(model_path, subfolder="controlnet")
    return ControlNetModel.from_pretrained(model_path)


def save_trainable_models(unet, controlnet, output_dir):
    unet.save_pretrained(os.path.join(output_dir, "unet"))
    controlnet.save_pretrained(os.path.join(output_dir, "controlnet"))


def log_validation(vae, unet, controlnet, args, accelerator, weight_dtype, step):
    if not args.validation_blur:
        return

    logger.info("Running deblur baseline validation...")
    scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=scheduler,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline._init_tiled_vae(
        encoder_tile_size=args.encoder_tiled_size,
        decoder_tile_size=args.decoder_tiled_size,
    )

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_dir = Path(args.output_dir) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    for image_index, image_path in enumerate(args.validation_blur):
        blur_image = Image.open(image_path).convert("RGB")
        original_size = blur_image.size
        width, height = blur_image.size
        width, height = width // 8 * 8, height // 8 * 8
        blur_image = blur_image.resize((width, height), Image.BICUBIC)

        for sample_index in range(args.num_validation_images):
            with torch.autocast(accelerator.device.type, enabled=accelerator.device.type == "cuda"):
                image = pipeline(
                    args,
                    image=blur_image,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    conditioning_scale=args.conditioning_scale,
                ).images[0]
            image = image.resize(original_size, Image.BICUBIC)
            image.save(validation_dir / f"step-{step:08d}-{image_index:02d}-{sample_index:02d}.png")


def main(args):
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN is disabled for this run.")

    UNet2DConditionModel, ControlNetModel = get_model_classes()
    log_with = None if args.report_to.lower() in {"none", "no", "disabled"} else args.report_to

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained_orig(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    cross_attention_dim = get_cross_attention_dim(unet)

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = load_controlnet(ControlNetModel, args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            for model in models:
                unwrapped_model = accelerator.unwrap_model(model)
                if isinstance(unwrapped_model, UNet2DConditionModel):
                    unwrapped_model.save_pretrained(os.path.join(output_dir, "unet"))
                elif isinstance(unwrapped_model, ControlNetModel):
                    unwrapped_model.save_pretrained(os.path.join(output_dir, "controlnet"))
            while len(weights) > 0:
                weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                unwrapped_model = accelerator.unwrap_model(model)
                if isinstance(unwrapped_model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                elif isinstance(unwrapped_model, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                else:
                    continue

                unwrapped_model.register_to_config(**load_model.config)
                unwrapped_model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    unet.train()

    for name, param in unet.named_parameters():
        if any(trainable_module_name in name for trainable_module_name in args.trainable_modules):
            param.requires_grad = True

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may be unstable for training; update xFormers if issues appear.")
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError("ControlNet must start in float32 precision.")
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError("UNet must start in float32 precision.")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes to use --use_8bit_adam.") from exc
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(controlnet.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = DeblurPairDataset(
        blur_dir=args.train_blur_dir,
        sharp_dir=args.train_sharp_dir,
        image_size=args.resolution,
        center_crop=args.center_crop,
        random_flip=not args.no_random_flip,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if len(train_dataloader) == 0:
        raise ValueError(
            "The training dataloader is empty. Reduce --train_batch_size or add more paired training images."
        )

    if args.max_train_steps <= 0:
        raise ValueError("`--max_train_steps` must be a positive integer.")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        controlnet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    params_to_clip = list(controlnet.parameters()) + [p for p in unet.parameters() if p.requires_grad]

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process and log_with is not None:
        tracker_config = dict(vars(args))
        tracker_config.pop("trainable_modules")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running deblur baseline training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
        else:
            checkpoint_step = int(path.split("-")[1])
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except KeyError as exc:
                if exc.args != ("step",):
                    raise
                accelerator.print(
                    "Checkpoint state does not contain Accelerator.step; "
                    "continuing after loading model, optimizer, scheduler, and dataloader states."
                )
                accelerator.step = checkpoint_step
            global_step = checkpoint_step
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            if global_step >= args.max_train_steps:
                accelerator.print(
                    f"Checkpoint step {global_step} has already reached --max_train_steps={args.max_train_steps}."
                )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet), accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )
                conditioning_pixel_values = batch["conditioning_pixel_values"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=True,
                )

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = build_null_encoder_hidden_states(
                    cross_attention_dim,
                    batch_size=bsz,
                    device=latents.device,
                    dtype=weight_dtype,
                )

                controlnet_cond_mid, down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if controlnet_cond_mid is not None:
                    if isinstance(controlnet_cond_mid, list):
                        for values in controlnet_cond_mid:
                            target_image = F.interpolate(
                                pixel_values,
                                size=values.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                            loss += F.l1_loss(target_image.float(), values.float(), reduction="mean")
                    else:
                        loss += F.l1_loss(pixel_values.float(), controlnet_cond_mid.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_blur and global_step % args.validation_steps == 0:
                        log_validation(vae, unet, controlnet, args, accelerator, weight_dtype, global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_trainable_models(
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(controlnet),
            args.output_dir,
        )
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of deblur baseline training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
