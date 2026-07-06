from typing import Any

import torch


def get_cross_attention_dim(model_or_config: Any) -> int:
    if isinstance(model_or_config, int):
        return model_or_config
    config = getattr(model_or_config, "config", model_or_config)
    cross_attention_dim = getattr(config, "cross_attention_dim", None)
    if isinstance(cross_attention_dim, (list, tuple)):
        cross_attention_dim = cross_attention_dim[0]
    if cross_attention_dim is None:
        raise ValueError("Cannot infer cross_attention_dim from the model config.")
    return int(cross_attention_dim)


def build_null_encoder_hidden_states(
    model_or_config: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    sequence_length: int = 77,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = False,
) -> torch.Tensor:
    hidden_size = get_cross_attention_dim(model_or_config)
    encoder_hidden_states = torch.zeros(
        batch_size,
        sequence_length,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    if num_images_per_prompt > 1:
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
    if do_classifier_free_guidance:
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
    return encoder_hidden_states
