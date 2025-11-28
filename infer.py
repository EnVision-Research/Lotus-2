#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, set_peft_model_state_dict
from PIL import Image
from torch import nn
from tqdm.auto import tqdm

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.utils import  convert_unet_state_dict_to_peft
from utils.image_utils import colorize_depth_map
from pipeline import Lotus2Pipeline
from utils.seed_all import seed_all

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")


class Local_Continuity_Module(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.lcm = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        lcm_dtype = next(self.lcm.parameters()).dtype
        if x.dtype != lcm_dtype:
            x = x.to(dtype=lcm_dtype)
        return x + self.lcm(x)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Run Lotus-2.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--core_predictor_model_path",
        type=str,
        default=None,
        help="Path to core predictor model weights",
    )
    parser.add_argument(
        "--lcm_model_path",
        type=str,
        default=None,
        help="Path to local continuity module model weights",
    )
    parser.add_argument(
        "--detail_sharpener_model_path",
        type=str,
        default=None,
        help="Path to detail sharpener model weights",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="Number of timesteps to infer the model.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="The directory where the input images are stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def process_single_image(image_path, pipeline, task_name, device, 
                         num_inference_steps, process_res=None):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32)
    image_ts = torch.tensor(image_np).permute(2,0,1).unsqueeze(0)
    image_ts = image_ts / 127.5 - 1.0 
    image_ts = image_ts.to(device)

    height, width = image_ts.shape[2:]
    max_edge = max(height, width)
    if max_edge > 1024:
        process_res = 1024
    elif max_edge < 512:
        process_res = 512
    else:
        process_res = None

    prediction = pipeline(
        rgb_in=image_ts, 
        prompt='', 
        num_inference_steps=num_inference_steps,
        output_type='np',
        process_res=process_res,
        ).images[0]
    
    if task_name == "depth":
        output_npy = prediction.mean(axis=-1)
        output_vis = colorize_depth_map(output_npy, reverse_color=True)
    elif task_name == "normal":
        output_npy = prediction
        output_vis = Image.fromarray((output_npy * 255).astype(np.uint8))
    else:
        raise ValueError(f"Invalid task name: {task_name}")
        
    return image, output_vis, output_npy

def load_lora_and_lcm_weights(transformer, core_predictor_model_path, lcm_model_path, detail_sharpener_model_path, task_name):
    lora_rank = 128 if task_name == 'depth' else 256
    device = transformer.device
    weight_dtype = transformer.dtype

    target_lora_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]

    # load lora weights for core predictor
    core_transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_lora_modules,
    )
    transformer.add_adapter(core_transformer_lora_config, adapter_name="core_predictor")

    core_lora_state_dict = Lotus2Pipeline.lora_state_dict(core_predictor_model_path)
    core_transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v for k, v in core_lora_state_dict.items() if k.startswith("transformer.")
    }
    core_transformer_state_dict = convert_unet_state_dict_to_peft(core_transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, core_transformer_state_dict, adapter_name="core_predictor")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logging.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )

    for name, param in transformer.named_parameters():
        if "core_predictor" in name:
            param.requires_grad = False
    # transformer.to(device=device, dtype=weight_dtype)
    logging.info(f"Successfully loaded lora weights for [core predictor].")

    # stage1 lcm weights
    local_continuity_module = Local_Continuity_Module(transformer.config.in_channels//4)
    lcm_state_dict = torch.load(lcm_model_path, map_location="cpu", weights_only=True)
    local_continuity_module.load_state_dict(lcm_state_dict)
    local_continuity_module.requires_grad_(False)
    local_continuity_module.to(device=device, dtype=weight_dtype)
    logging.info(f"Successfully loaded weights for [local continuity module (LCM)].")

    # stage2 lora weights (detail sharpener)
    sharpener_transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_lora_modules,
    )
    transformer.add_adapter(sharpener_transformer_lora_config, adapter_name="detail_sharpener")

    sharpener_lora_state_dict = Lotus2Pipeline.lora_state_dict(detail_sharpener_model_path)
    sharpener_transformer_state_dict = {
        f'{k.replace("transformer.", "")}': v for k, v in sharpener_lora_state_dict.items() if k.startswith("transformer.")
    }
    sharpener_transformer_state_dict = convert_unet_state_dict_to_peft(sharpener_transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, sharpener_transformer_state_dict, adapter_name="detail_sharpener")
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logging.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )

    # freeze the stage2 lora
    for name, param in transformer.named_parameters():
        if "detail_sharpener" in name:
            param.requires_grad = False
    # transformer.to(device=device, dtype=weight_dtype)
    logging.info(f"Successfully loaded lora weights for [detail sharpener].")

    return transformer, local_continuity_module

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Run Lotus-2! ")

    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)

    output_dir_vis = os.path.join(args.output_dir, f'{args.task_name}_vis')
    output_dir_npy = os.path.join(args.output_dir, f'{args.task_name}_npy')
    if not os.path.exists(output_dir_vis): os.makedirs(output_dir_vis)
    if not os.path.exists(output_dir_npy): os.makedirs(output_dir_npy)

    logging.info(f"Output dir = {args.output_dir}")

    # Mixed precision
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32  
    logging.info(f"Running with {weight_dtype} precision.")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    input_dir = Path(args.input_dir)
    test_images = list(input_dir.rglob('*.png')) + list(input_dir.rglob('*.jpg'))
    test_images = sorted(test_images)
    logging.info(f'==> There are {len(test_images)} images for validation.')

    # -------------------- Load scheduler and models --------------------
    # scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", num_train_timesteps=10
    )
    # transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)
    
    # load weights
    transformer, local_continuity_module = load_lora_and_lcm_weights(transformer, 
                                            args.core_predictor_model_path, 
                                            args.lcm_model_path,
                                            args.detail_sharpener_model_path,
                                            args.task_name
                                            )

    # -------------------- Pipeline --------------------
    pipeline = Lotus2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=noise_scheduler,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.local_continuity_module = local_continuity_module
    pipeline = pipeline.to(device)
    
    # -------------------- Run inference! --------------------
    pipeline.set_progress_bar_config(disable=True)

    with nullcontext():
        for image_path in tqdm(test_images):
            # print("\n",image_path)
            _, output_vis, output_npy = process_single_image(
                image_path, pipeline, 
                task_name=args.task_name,
                device=device,
                num_inference_steps=args.num_inference_steps,
            )
            
            output_vis.save(os.path.join(output_dir_vis, f'{image_path.stem}.png'))
            np.save(os.path.join(output_dir_npy, f'{image_path.stem}.npy'), output_npy)

if __name__ == "__main__":
    args = parse_args()
    main(args)
