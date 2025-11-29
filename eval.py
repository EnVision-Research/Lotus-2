import sys
import os
import torch
from PIL import Image
import gradio as gr
from glob import glob
from contextlib import nullcontext
from pipeline import Lotus2Pipeline
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from infer import (
    load_lora_and_lcm_weights,
    process_single_image
)
from evaluation.evaluation import evaluation_depth, evaluation_normal


pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.bfloat16
task = os.environ.get("TASK_NAME", "depth") # or normal

def load_pipeline():
    global pipeline, device, weight_dtype, task
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder="scheduler", num_train_timesteps=10
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        'black-forest-labs/FLUX.1-dev', subfolder="transformer", revision=None, variant=None
    )
    transformer.requires_grad_(False)
    transformer.to(device=device, dtype=weight_dtype)
    transformer, local_continuity_module = load_lora_and_lcm_weights(transformer, None, None, None, task)
    pipeline = Lotus2Pipeline.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        scheduler=noise_scheduler,
        transformer=transformer,
        revision=None,
        variant=None,
        torch_dtype=weight_dtype,
    )
    pipeline.local_continuity_module = local_continuity_module
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

def eval():
    global pipeline, device, weight_dtype, task
    base_test_data_dir = os.environ.get("TEST_DATA_DIR", "datasets/eval")
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/eval")

    def gen_fn(rgb_in):
        if task == "depth":
            rgb_input = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            output_type = "np"
        elif task == "normal":
            rgb_input = rgb_in
            output_type = "pt"
        else:
            raise ValueError(f"Invalid task name: {task}")

        prediction = pipeline(
            rgb_in=rgb_input, 
            prompt='', 
            num_inference_steps=10,
            output_type=output_type,
            process_res=None
            ).images[0]

        if task == "depth":
            output = prediction.mean(axis=-1)
        elif task == "normal":
            output = (prediction * 2.0 - 1.0).unsqueeze(0) # [0,1] -> [-1,1], (1, 3, h, w)
        return output

    with torch.no_grad():
        if task == 'depth':
            test_data_dir = os.path.join(base_test_data_dir, task)
            test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
                "kitti": "configs/data_kitti_eigen_test.yaml",
                "scannet": "configs/data_scannet_val.yaml",
                "eth3d": "configs/data_eth3d.yaml",
                "diode": "configs/data_diode_all.yaml",
            }
            for dataset_name, config_path in test_depth_dataset_configs.items():
                eval_dir = os.path.join(output_dir, task, dataset_name)
                test_dataset_config = os.path.join(test_data_dir, config_path)
                alignment_type = "least_square_disparity"
                metric_tracker = evaluation_depth(eval_dir, test_dataset_config, test_data_dir, eval_mode="generate_prediction",
                                                  gen_prediction=gen_fn, pipeline=pipeline, alignment=alignment_type, processing_res=None)
                print(dataset_name,',', 'abs_relative_difference: ', metric_tracker.result()['abs_relative_difference'], 'delta1_acc: ', metric_tracker.result()['delta1_acc'])
        elif task == 'normal':
            test_data_dir = os.path.join(base_test_data_dir, task)
            dataset_split_path = "evaluation/dataset_normal"
            eval_datasets = [ ('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel'),  ('oasis', 'val')]
            eval_dir = os.path.join(output_dir, task)
            evaluation_normal(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                              gen_prediction=gen_fn, pipeline=pipeline, eval_datasets=eval_datasets, processing_res=None)
        else:
            raise ValueError(f"Not support predicting {task} yet. ")
        
        print('==> Evaluation is done. \n==> Results saved to:', output_dir)


if __name__ == "__main__":
    load_pipeline()
    eval()
