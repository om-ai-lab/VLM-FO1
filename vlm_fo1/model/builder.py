from transformers import AutoTokenizer
import torch
from vlm_fo1.model import *
from safetensors.torch import load_file
import os


def load_pretrained_model(model_path, load_8bit=False, load_4bit=False, device="cuda"):
    """
    Loads a pretrained model along with its vision towers (and associated image processors).
    This function supports loading in 8bit/4bit precision and explicit device placement.

    Args:
        model_path (str): Path to the pretrained model directory.
        load_8bit (bool): Whether to load the model in 8bit mode.
        load_4bit (bool): Whether to load the model in 4bit mode.
        device (str): Device to load model onto, e.g., "cuda" or "cpu".

    Returns:
        tuple: (tokenizer, model, image_processor)
    """
    kwargs = {"device_map": device}

    # Set model loading parameters for quantization or floating point
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    # print(model_path)

    # Only proceed for vlm-fo1 models
    if 'vlm-fo1' in model_path.lower():
        # Load tokenizer (slow tokenizer enforced)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # If this is the Qwen2.5-VL variant, load with additional kwargs
        if 'qwen2.5-vl' in model_path.lower() or 'qwen2_5_vl' in model_path.lower() or "vlm-fo1-3b" in model_path.lower():
            model, loading_info = OmChatQwen25VLForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                output_loading_info=True,
                attn_implementation="flash_attention_2",
                **kwargs
            )
            # print(f'OmChatQwen25VLForCausalLM loading_info: {loading_info}')
        # (For other variants of vlm-fo1, model loading detail may need additional condition.)

    if 'vlm-fo1' in model_path.lower():
        # --- Vision Tower Loading ---
        # Load the main vision tower weights from model_path if it is not yet loaded
        primary_vision_tower = model.get_vision_tower()
        if primary_vision_tower and not primary_vision_tower.is_loaded:
            primary_vision_tower.load_model(model_path=model_path, is_train=False)
            primary_vision_tower.to(device=device, dtype=torch.bfloat16)  # Move to correct device/dtype

        # Grab primary image processor from vision tower, if present
        if primary_vision_tower:
            primary_image_processor = primary_vision_tower.image_processor

        # --- Auxiliary Vision Tower Handling (Qwen2.5-VL case only) ---
        if 'qwen2.5-vl' in model_path.lower() or 'qwen2_5_vl' in model_path.lower() or "vlm-fo1-3b" in model_path.lower():
            try:
                aux_image_size = model.config.aux_image_size
            except Exception:
                # If aux_image_size is missing from config fallback to 768
                aux_image_size = 768

            aux_image_aspect_ratio = model.config.aux_image_aspect_ratio
            aux_vision_tower = model.get_vision_tower_aux()
            # Only load if not already loaded
            if aux_vision_tower and not aux_vision_tower.is_loaded:
                aux_vision_tower.load_model(image_size=aux_image_size, is_train=False, aspect_ratio=aux_image_aspect_ratio)
                aux_vision_tower.to(device=device, dtype=torch.bfloat16)

        # Get auxiliary image processor if there is an aux vision tower
        if aux_vision_tower:
            aux_image_processor = aux_vision_tower.image_processor
        else:
            image_processor = None  # Set to None if there is no auxiliary vision tower

        # image_processor returned as a tuple of (primary, aux)
        image_processor = (primary_image_processor, aux_image_processor)

    # Set model to eval mode and move to correct device before returning
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return tokenizer, model, image_processor
