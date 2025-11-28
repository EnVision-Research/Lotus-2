from typing import Union, Optional, List, Dict, Any
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from diffusers.utils import is_torch_xla_available

from utils.image_utils import resize_image, resize_image_first

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class Lotus2Pipeline(FluxPipeline):
    @torch.no_grad()
    def __call__(
        self,
        rgb_in: Optional[torch.FloatTensor] = None,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 10,
        output_type: Optional[str] = "pil",
        process_res: Optional[int] = None,
        timestep_core_predictor: int = 1,
        guidance_scale: float = 3.5,
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            rgb_in (`torch.FloatTensor`, *optional*):
                The input image to be used for generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the prediction. Default is ''. 
            num_inference_steps (`int`, *optional*, defaults to 10):
                The number of denoising steps. More denoising steps usually lead to a sharper prediction at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        # 1. prepare
        batch_size = rgb_in.shape[0]
        input_size = rgb_in.shape[2:]
        rgb_in = resize_image_first(rgb_in, process_res)
        height, width = rgb_in.shape[2:]

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        device = self._execution_device

        # 2. encode prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
        )

        # 3. prepare latent variables
        rgb_in = rgb_in.to(device=device, dtype=self.dtype)
        rgb_latents = self.vae.encode(rgb_in).latent_dist.sample()
        rgb_latents = (rgb_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        packed_rgb_latents = self._pack_latents(
            rgb_latents,
            batch_size=rgb_latents.shape[0],
            num_channels_latents=rgb_latents.shape[1],
            height=rgb_latents.shape[2],
            width=rgb_latents.shape[3],
        )        
        
        latent_image_ids_core_predictor = self._prepare_latent_image_ids(batch_size, rgb_latents.shape[2]//2, rgb_latents.shape[3]//2, device, rgb_latents.dtype)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, rgb_latents.shape[2]//2, rgb_latents.shape[3]//2, device, rgb_latents.dtype)

        # 4. prepare timesteps
        timestep_core_predictor = torch.tensor(timestep_core_predictor).expand(batch_size).to(device=rgb_in.device, dtype=rgb_in.dtype)

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = packed_rgb_latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0) # 0
        self._num_timesteps = len(timesteps)

        # 5. handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(packed_rgb_latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        # 6. core predictor
        self.transformer.set_adapter("core_predictor")
        latents = self.transformer(
            hidden_states=packed_rgb_latents,
            timestep=timestep_core_predictor / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids_core_predictor,
            joint_attention_kwargs=self.joint_attention_kwargs, # {}
            return_dict=False,
        )[0]
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = self.local_continuity_module(latents)
        
        # 7. Denoising loop for detail sharpener
        self.transformer.set_adapter("detail_sharpener")
        latents = self._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[2],
            width=latents.shape[3],
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        latents = latents.to(dtype=self.dtype)

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Resize output image to match input size
        image = resize_image(image, input_size)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)