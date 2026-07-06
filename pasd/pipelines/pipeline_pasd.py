# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor

from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from ..myutils.vaehook import VAEHook, perfcount
from ..myutils.null_condition import build_null_encoder_hidden_states


class StableDiffusionControlNetPipeline(DiffusionPipeline):
    r"""
    No-text PASD deblurring pipeline.

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        unet:
            Conditional U-Net architecture to denoise the encoded image latents.
        controlnet:
            Image-conditioned ControlNet used to guide deblurring.
        scheduler ([`SchedulerMixin`]):
            Scheduler used with `unet` to denoise the encoded image latents.

    This repository removes the tokenizer, CLIP text encoder, prompt handling, and textual inversion paths. The UNet
    and ControlNet still receive a zero-valued cross-attention tensor so the Stable Diffusion architecture stays shape
    compatible with SD2-base checkpoints.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: Any,
        controlnet: Any,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _init_tiled_vae(self,
            encoder_tile_size = 1024,
            decoder_tile_size = 256,
            fast_decoder = False,
            fast_encoder = False,
            color_fix = False,
            vae_to_gpu = True):
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet, vae, and
        controlnet have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.vae, self.controlnet]:
            if cpu_offloaded_model is None:
                continue
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.unet, self.vae]:
            if cpu_offloaded_model is None:
                continue
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_no_text_condition(
        self,
        device,
        batch_size,
        num_images_per_input,
    ):
        return build_null_encoder_hidden_states(
            self.unet,
            batch_size=batch_size,
            device=device,
            dtype=self.unet.dtype,
            num_images_per_input=num_images_per_input,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        #extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        image,
        height,
        width,
        callback_steps,
        conditioning_scale=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        self.check_image(image)
        if not isinstance(conditioning_scale, (float, int)):
            raise TypeError("`conditioning_scale` must be a number for the deblurring baseline.")

    def check_image(self, image):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_pil_list = isinstance(image, list) and len(image) > 0 and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and len(image) > 0 and isinstance(image[0], torch.Tensor)

        if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
            raise TypeError(
                "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
            )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_input,
        device,
        dtype,
    ):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)#.flip(1)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_input

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, args, image, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if args is None or args.init_latent_with_noise:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                offset_noise = torch.randn(batch_size, num_channels_latents, 1, 1, device=device).to(dtype)
                offset_noise_scale = args.offset_noise_scale if args is not None else 0.0
                latents = latents + offset_noise_scale * offset_noise
            else:
                latents = latents.to(device)
        else:
            #print(image.shape, image.min(), image.max())
            if dtype==torch.float16: self.vae.quant_conv.half()
            init_latents = self.vae.encode(image*2.0-1.0).latent_dist.sample(generator)
            init_latents = self.vae.config.scaling_factor * init_latents
            self.scheduler.set_timesteps(args.num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps[0:]
            latent_timestep = timesteps[:1].repeat(batch_size * 1) # 999 what if <999"?
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)

            added_latent_timestep = torch.LongTensor([args.added_noise_level]).repeat(batch_size * 1).to(self.device)
            added_noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            init_latents = self.scheduler.add_noise(init_latents, added_noise, added_latent_timestep)

            latents = init_latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _default_height_width(self, height, width, image):
        # NOTE: It is possible that a list of images have different
        # dimensions for each image, so just checking the first image
        # is not _exactly_ correct, but it is simple.
        while isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def _infer_image_batch_size(self, image):
        if isinstance(image, PIL.Image.Image):
            return 1
        if isinstance(image, torch.Tensor):
            return image.shape[0]
        if isinstance(image, list):
            return len(image)
        return 1

    # override DiffusionPipeline
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
    ):
        super().save_pretrained(save_directory, safe_serialization, variant)
        
    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))

    @perfcount
    @torch.no_grad()
    def __call__(
        self,
        args = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_images_per_input: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        conditioning_scale: float = 1.0,
    ):
        r"""
        Run no-text image deblurring.

        `image` is the blurred image condition. The pipeline always uses a fixed zero cross-attention condition instead
        of tokenizer/CLIP text features.
        """
        height, width = self._default_height_width(height, width, image)
        self.check_inputs(
            image,
            height,
            width,
            callback_steps,
            conditioning_scale,
        )

        batch_size = self._infer_image_batch_size(image)
        effective_batch_size = batch_size * num_images_per_input
        device = self._execution_device
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        encoder_hidden_states = self._encode_no_text_condition(
            device,
            batch_size,
            num_images_per_input,
        )

        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=effective_batch_size,
            num_images_per_input=num_images_per_input,
            device=device,
            dtype=controlnet.dtype,
        )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            args,
            image[:1],
            effective_batch_size,
            num_channels_latents,
            height,
            width,
            encoder_hidden_states.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                controlnet_latent_model_input = latent_model_input

                _, _, h, w = latent_model_input.size()
                tile_size, tile_overlap = (args.latent_tiled_size, args.latent_tiled_overlap) if args is not None else (256, 8)
                if h*w<=tile_size*tile_size: #h<tile_size and w<tile_size: # tiled latent input
                    down_block_res_samples, mid_block_res_sample = [None]*10, None
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        controlnet_latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=image,
                        conditioning_scale=conditioning_scale,
                        return_dict=False,
                    )

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]
                else:
                    tile_size = min(tile_size, min(h, w))
                    tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

                    grid_rows = 0
                    cur_x = 0
                    while cur_x < latent_model_input.size(-1):
                        cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                        grid_rows += 1

                    grid_cols = 0
                    cur_y = 0
                    while cur_y < latent_model_input.size(-2):
                        cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                        grid_cols += 1

                    input_list = []
                    cond_list = []
                    img_list = []
                    noise_preds = []
                    for row in range(grid_rows):
                        noise_preds_row = []
                        for col in range(grid_cols):
                            if col < grid_cols-1 or row < grid_rows-1:
                                # extract tile from input image
                                ofs_x = max(row * tile_size-tile_overlap * row, 0)
                                ofs_y = max(col * tile_size-tile_overlap * col, 0)
                                # input tile area on total image
                            if row == grid_rows-1:
                                ofs_x = w - tile_size
                            if col == grid_cols-1:
                                ofs_y = h - tile_size

                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size

                            # input tile dimensions
                            input_tile = latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            input_list.append(input_tile)
                            cond_tile = controlnet_latent_model_input[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                            cond_list.append(cond_tile)
                            img_tile = image[:, :, input_start_y*8:input_end_y*8, input_start_x*8:input_end_x*8]
                            img_list.append(img_tile)

                            if len(input_list) == batch_size or col == grid_cols-1:
                                input_list_t = torch.cat(input_list, dim=0)
                                cond_list_t = torch.cat(cond_list, dim=0)
                                img_list_t = torch.cat(img_list, dim=0)
                                #print(input_list_t.shape, cond_list_t.shape, img_list_t.shape, fg_mask_list_t.shape)

                                down_block_res_samples, mid_block_res_sample = self.controlnet(
                                    cond_list_t,
                                    t,
                                    encoder_hidden_states=encoder_hidden_states,
                                    controlnet_cond=img_list_t,
                                    conditioning_scale=conditioning_scale,
                                    return_dict=False,
                                )

                                # predict the noise residual
                                model_out = self.unet(
                                    input_list_t,
                                    t,
                                    encoder_hidden_states=encoder_hidden_states,
                                    cross_attention_kwargs=cross_attention_kwargs,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample,
                                    return_dict=False,
                                )[0]

                                #for sample_i in range(model_out.size(0)):
                                #    noise_preds_row.append(model_out[sample_i].unsqueeze(0))
                                input_list = []
                                cond_list = []
                                img_list = []

                            noise_preds.append(model_out)

                    # Stitch noise predictions for all tiles
                    noise_pred = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
                    contributors = torch.zeros(latent_model_input.shape, device=latent_model_input.device)
                    # Add each tile contribution to overall latents
                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if col < grid_cols-1 or row < grid_rows-1:
                                # extract tile from input image
                                ofs_x = max(row * tile_size-tile_overlap * row, 0)
                                ofs_y = max(col * tile_size-tile_overlap * col, 0)
                                # input tile area on total image
                            if row == grid_rows-1:
                                ofs_x = w - tile_size
                            if col == grid_cols-1:
                                ofs_y = h - tile_size

                            input_start_x = ofs_x
                            input_end_x = ofs_x + tile_size
                            input_start_y = ofs_y
                            input_end_y = ofs_y + tile_size
    
                            noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                            contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                    # Average overlapping areas with more than 1 contributor
                    noise_pred /= contributors

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0].to(encoder_hidden_states.dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        has_nsfw_concept = None
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]#.flip(1)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
