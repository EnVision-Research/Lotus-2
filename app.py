import spaces  # must be first!
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

pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_dtype = torch.bfloat16
task = None

@spaces.GPU
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

@spaces.GPU
def fn(image_path):
    global pipeline, device, task
    pipeline.set_progress_bar_config(disable=True)
    with nullcontext():
        _, output_vis, _ = process_single_image(
            image_path, pipeline, 
            task_name=task,
            device=device,
            num_inference_steps=10,
            process_res=1024
        )
    return [Image.open(image_path), output_vis]

def build_demo():
    global task
    inputs = [
        gr.Image(label="Image", type="filepath")
    ]
    outputs = [
        gr.ImageSlider(
            label=f"{task.title()}",
            type="pil",
            slider_position=20,
        )
    ]
    examples = glob(f"assets/demo_examples/{task}/*.png") + glob(f"assets/demo_examples/{task}/*.jpg")
    demo = gr.Interface(
        fn=fn,
        title="Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model",
        description=f"""
            <strong>Please consider starring <span style="color: orange">&#9733;</span> our <a href="https://github.com/EnVision-Research/Lotus-2" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this demo useful! 😊</strong>
            <br>
            <strong>Current Task: </strong><strong style="color: red;">{task.title()}</strong>
        """,
        inputs=inputs,
        outputs=outputs,
        examples=examples,
        examples_per_page=10
    )
    return demo

def main(task_name):
    global task
    task = task_name
    load_pipeline()
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=6381,
    )

if __name__ == "__main__":
    task_name = sys.argv[-1]
    if not task_name in ['depth', 'normal']:
        raise ValueError("Invalid task. Please choose from 'depth' and 'normal'.")
    main(task_name)
