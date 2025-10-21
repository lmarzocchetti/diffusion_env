import argparse
import os
import sys
import time
import random
from pathlib import Path
from PIL import Image

import scipy.ndimage
import torch
import Imath
import OpenEXR
import numpy as np
import cv2
import scipy

# import matplotlib.pyplot as plt
from diffusers import AutoPipelineForInpainting
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import *

# import 
target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ext", "Text2Light"))
sys.path.append(target_dir)
from sritmo.global_sritmo import SRiTMO

class myStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super(myStableDiffusionImg2ImgPipeline, self).__init__(
            vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder, requires_safety_checker
        )

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # MASK
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        latents: Optional[torch.FloatTensor] = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        image = init_image.to(dtype=torch.float32)

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        is_strength_max = strength == 1.0

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 6.1 Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 6.2 Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

def load_exr(filename):
    """Load an EXR image and return it as a NumPy array."""
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header['channels'].keys())  # Convert to a list
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read channels
    img = {ch: np.frombuffer(exr_file.channel(ch, pixel_type), dtype=np.float32)
           .reshape(height, width) for ch in channels}

    return img, width, height, channels

def resize_exr(img, new_width, new_height):
    """Resize an EXR image using OpenCV."""
    resized_img = {ch: cv2.resize(img[ch], (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
                   for ch in img}
    return resized_img

def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)

def change_luminance(rgb: np.ndarray, l_out: np.ndarray) -> np.ndarray:
    l_in: np.ndarray = luminance(rgb)
    return rgb * np.expand_dims((l_out / l_in), axis = -1)

def rgb_to_ext_reinhard(rgb: np.ndarray, max_white_l=100) -> np.ndarray:
    l_old = luminance(rgb)
    numerator = l_old * (1.0 + (l_old / (max_white_l * max_white_l)))
    l_new = numerator / (1.0 + l_old)

    return change_luminance(rgb, l_new)

def reinhard_to_rgb(reinhard: np.ndarray, white_lum=100) -> np.ndarray:
    new_lum = luminance(reinhard)

    # Solve the quadratic equation
    a = 1.0 / white_lum ** 2
    b = 1.0 - new_lum
    c = -new_lum

    discriminant = np.sqrt(b ** 2 - 4 * a * c)
    lum = (-b + discriminant) / (2 * a)

    # Scale reinhard_rgb back to original rgb
    return reinhard * np.where(new_lum != 0, (lum / new_lum), 0)[:, :, None]

def rgb_to_hlg(rgb: np.ndarray) -> np.ndarray:
    hlg = None
    with np.errstate(divide='ignore', invalid='ignore'):
        hlg = np.where(rgb <= 1.0,
                    0.5 * np.sqrt(rgb),
                    0.17883277 * np.log(rgb - 0.28466892) + 0.55991073)
        hlg[np.isinf(hlg)] = 0
        hlg[np.isnan(hlg)] = 0
    return hlg


def hlg_to_rgb(hlg: np.ndarray) -> np.ndarray:
    rgb = None
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.where(hlg <= 0.5,
                    np.square(2.0 * hlg),
                    np.exp((hlg - 0.55991073) / 0.17883277) + 0.28466892)
        rgb[np.isinf(rgb)] = 0
        rgb[np.isnan(rgb)] = 0
    return rgb

def pq_to_rgb(rgb: np.ndarray, mul: int = 10000) -> np.ndarray:
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ret = None
    with np.errstate(divide='ignore', invalid='ignore'):
        E_p = rgb ** (1/m2)
        par = np.maximum((E_p - c1), 0) / (c2 - c3 * E_p)
        ret = mul * (par ** (1/m1))
        ret[np.isinf(ret)] = 0
        ret[np.isnan(ret)] = 0
    
    return np.clip(ret, 0, 1)

def rgb_to_pq(pq: np.ndarray, div: int = 10000) -> np.ndarray:
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    ret = None
    with np.errstate(divide='ignore', invalid='ignore'):
        Y = pq / div
        ret = ((c1 + c2 * (Y ** m1)) / (1 + c3 * (Y ** m1))) ** m2
        ret[np.isinf(ret)] = 0
        ret[np.isnan(ret)] = 0

    return ret

def save_exr(img, filename):
    height, width, _ = img.shape
    header = OpenEXR.Header(width, height)
    exr = OpenEXR.OutputFile(filename, header)

    r, g, b = np.split(img, 3, axis=-1)
    exr.writePixels({'R': r.tobytes(),
	                 'G': g.tobytes(),
	                 'B': b.tobytes()})
    exr.close()

def set_seed(seed = None):
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_argument():
    parser = argparse.ArgumentParser(description="Inpaint a batch of exr images")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to the finetuned LoRA version"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        required=True,
        help="Version of stable diffusion to use",
        choices=['sd2', 'sdxl', 'img2img']
    )
    parser.add_argument(
        "--test_input_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the directory with directories having ([name_of_the_folder]_input.exr mask_1_sky.png mask_2_light.png prompt.txt). In every directory the output file will be saved as output.exr"
    )
    parser.add_argument(
        "--test_output_dir",
        type=str,
        default="test_output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--color_space",
        type=str,
        default=None,
        required=True,
        choices=['reinhard', 'hlg', 'pq']
    )

    args = parser.parse_args()
    return args

def choose_color_space_function(color_space: str):
    to_fn = None
    from_fn = None
    match color_space:
        case 'reinhard':
            to_fn = rgb_to_ext_reinhard
            from_fn = reinhard_to_rgb
        case 'hlg':
            to_fn = rgb_to_hlg
            from_fn = hlg_to_rgb
        case 'pq':
            to_fn = rgb_to_pq
            from_fn = pq_to_rgb
    
    return to_fn, from_fn

def parse_prompt_file(filename: str) -> tuple[str, str]:
    with open(filename) as f:
        lines = f.readlines()
        assert len(lines) == 2
        return lines[0], lines[1]

def normal_generation(
        pipe: AutoPipelineForInpainting | StableDiffusionInpaintPipeline, 
        input_image: np.ndarray, 
        mask_1_sky: Image.Image, 
        mask_2_light: Image.Image,
        prompt_1_sky: str,
        prompt_2_light: str,
        negative_prompt_1_sky: str,
        negative_prompt_2_light:str,
        color_space: str,
        from_fn) -> tuple[np.ndarray, np.ndarray]:
    image: Image.Image = pipe(prompt=prompt_1_sky, negative_prompt=negative_prompt_1_sky, image=input_image, mask_image=mask_1_sky, num_inference_steps=50, strength=0.7, guidance_scale=7.5, width=1024, height=512, output_type='np').images[0]
    image: Image.Image = pipe(prompt=prompt_2_light, negative_prompt=negative_prompt_2_light, image=image, mask_image=mask_2_light, num_inference_steps=100, strength=0.7, guidance_scale=7.5, width=1024, height=512, output_type='np').images[0]

    output_to_save = None
    match color_space:
        case 'reinhard':
            output_to_save = from_fn(image)
        case 'hlg':
            output_to_save = from_fn(image)
        case 'pq':
            output_to_save = from_fn(image, mul = 100)
    
    return output_to_save, image


def contrast_generation(
        pipe: AutoPipelineForInpainting | StableDiffusionInpaintPipeline, 
        input_image: np.ndarray, 
        mask_1_sky: Image.Image, 
        mask_2_light: Image.Image,
        prompt_1_sky: str,
        prompt_2_light: str,
        negative_prompt_1_sky: str,
        negative_prompt_2_light:str,
        color_space: str,
        from_fn,
        folder_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    mask_for_contrast = None
    if os.path.isfile(f"{folder_path}/mask_contrast.png"):
        mask_for_contrast = Image.open(f"{folder_path}/mask_contrast.png")
    else:
        mask_for_contrast = mask_2_light
    
    # TODO: remove this
    mask_for_contrast = mask_2_light

    #ci vuole la 2 
    mask_2_light_np = (np.array(mask_for_contrast.convert("L")) // 255)
    scaling_factor = random.random()
    # TODO: Exclude the center of the mask
    # TODO: verify if this works!
    input_image[mask_2_light_np == 1] *= scaling_factor

    image: Image.Image = pipe(prompt=prompt_1_sky, negative_prompt=negative_prompt_1_sky, image=input_image, mask_image=mask_1_sky, num_inference_steps=50, strength=0.7, guidance_scale=7.5, width=1024, height=512, output_type='np').images[0]
    mask_2_light_blurred = pipe.mask_processor.blur(mask_2_light, blur_factor=10).convert('RGB')
    image: Image.Image = pipe(prompt=prompt_2_light, negative_prompt=negative_prompt_2_light, image=image, mask_image=mask_2_light_blurred, num_inference_steps=100, strength=0.7, guidance_scale=7.5, width=1024, height=512, output_type='np').images[0]

    output_to_save = None
    match color_space:
        case 'reinhard':
            output_to_save = from_fn(image)
        case 'hlg':
            output_to_save = from_fn(image)
        case 'pq':
            output_to_save = from_fn(image, mul = 100)
    
    return output_to_save, image, scaling_factor


def blur_generation(
        pipe: AutoPipelineForInpainting | StableDiffusionInpaintPipeline, 
        input_image: np.ndarray, 
        mask_1_sky: Image.Image, 
        mask_2_light: Image.Image,
        prompt_1_sky: str,
        prompt_2_light: str,
        negative_prompt_1_sky: str,
        negative_prompt_2_light:str,
        color_space: str,
        from_fn) -> tuple[np.ndarray, np.ndarray]:
    mask_2_light_np = np.array(mask_2_light.convert("L")) // 255
    only_sun = input_image.copy() * np.expand_dims(mask_2_light_np, axis=-1)
    # only_sun_blurred = cv2.GaussianBlur(only_sun.astype(np.float32), (5, 5), 1)
    only_sun_blurred = scipy.ndimage.gaussian_filter(only_sun.astype(np.float32), sigma=3)
    # TODO: Usare versione scipy, sigma [3-5]
    # print(f"DEBUG: {only_sun_blurred.max()}")
    input_image[mask_2_light_np == 1] *= 0

    input_image = input_image + only_sun_blurred

    image: Image.Image = pipe(prompt=prompt_1_sky, negative_prompt=negative_prompt_1_sky, image=input_image, mask_image=mask_1_sky, num_inference_steps=50, strength=0.7, guidance_scale=7.5, width=1024, height=512, output_type='np').images[0]
    mask_2_light_blurred = pipe.mask_processor.blur(mask_2_light, blur_factor=10).convert('RGB')
    image: Image.Image = pipe(prompt=prompt_2_light, negative_prompt=negative_prompt_2_light, image=image, mask_image=mask_2_light_blurred, num_inference_steps=100, strength=0.7, guidance_scale=7.5, width=1024, height=512, output_type='np').images[0]

    output_to_save = None
    match color_space:
        case 'reinhard':
            output_to_save = from_fn(image)
        case 'hlg':
            output_to_save = from_fn(image)
        case 'pq':
            output_to_save = from_fn(image, mul = 100)
    
    return output_to_save, image

def main():
    set_seed(seed=42)

    args = parse_argument()
    to_fn, from_fn = choose_color_space_function(args.color_space)

    pipe = None
    if args.model_version == "sdxl":
        # SDXL
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            use_safetensors = True,
            safety_checker=None
        )
    elif args.model_version == "img2img":
        pipe = myStableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            use_safetensors = True,
            safety_checker=None
        )
    else:
        # SD2
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            # revision="fp16",
            torch_dtype=torch.float16,
            use_safetensors = True,
            safety_checker=None
        )
    pipe.unet.load_attn_procs(args.model_path)
    pipe.to('cuda')

    test_dir_path: Path = Path(args.test_input_dir)
    test_out_path: str = args.test_output_dir
    input_folders: list[str] = [f.name for f in test_dir_path.iterdir() if f.is_dir()]
    
    os.makedirs(args.test_output_dir, exist_ok=True)
    
    # Open the html file
    html_file = open(f"{test_out_path}/index_{args.color_space}_{args.model_version}.html", "w")
    html_file.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Basic Template</title>
</head>
<body>""")

    for folder in input_folders:
        print(f"--------------- START INPAINT {folder} ---------------")

        # Create output folder
        out_sub_folder = f"{test_out_path}/{folder}"
        os.makedirs(out_sub_folder, exist_ok=True)

        sub_path = f"{args.test_input_dir}/{folder}"
        input_image = f"{sub_path}/{folder}_input.exr"
        input_image_original = f"{sub_path}/{folder}_original.exr"
        mask_1_image = f"{sub_path}/mask_1_sky.png"
        mask_2_image = f"{sub_path}/mask_2_light.png"
        prompt_file = f"{sub_path}/prompt.txt"
        
        input_image_png = f"{out_sub_folder}/input.png"
        input_image_original_png = f"{out_sub_folder}/input_original.png"
        output_image = f"{out_sub_folder}/output_{args.color_space}_{args.model_version}.exr"
        output_image_png = f"{out_sub_folder}/output_{args.color_space}_{args.model_version}.png"
        output_image_contrast = f"{out_sub_folder}/output_contrast_{args.color_space}_{args.model_version}.exr"
        output_image_contrast_png = f"{out_sub_folder}/output_contrast_{args.color_space}_{args.model_version}.png"
        output_image_blur = f"{out_sub_folder}/output_blur_{args.color_space}_{args.model_version}.exr"
        output_image_blur_png = f"{out_sub_folder}/output_blur_{args.color_space}_{args.model_version}.png"
        output_mask_1_png = f"{out_sub_folder}/mask_1_sky.png"
        output_mask_2_png = f"{out_sub_folder}/mask_2_light.png"

        prompt_1_sky, prompt_2_light = parse_prompt_file(prompt_file)
        negative_prompt_1_sky, negative_prompt_2_light = "boxes, artifacts, light", "boxes, artifacts"
        print(f"prompt 1: {prompt_1_sky}---prompt 2: {prompt_2_light}")

        # Edited image
        img, width, height, _ = load_exr(input_image)
        new_width, new_height = int(width * 1.0), int(height * 1.0)
        resized_img = resize_exr(img, new_width, new_height)
        resized_rgb_image: np.ndarray = np.stack([resized_img['R'], resized_img['G'], resized_img['B']], axis=-1)

        # Original image
        img, width, height, _ = load_exr(input_image_original)
        new_width, new_height = int(width * 1.0), int(height * 1.0)
        resized_img = resize_exr(img, new_width, new_height)
        resized_rgb_image_orignal: np.ndarray = np.stack([resized_img['R'], resized_img['G'], resized_img['B']], axis=-1)
        
        out_color_space: np.ndarray = None
        out_color_space_original: np.ndarray = None
        match args.color_space:
            case 'reinhard':
                out_color_space = to_fn(resized_rgb_image)
                out_color_space_original = to_fn(resized_rgb_image_orignal)
            case 'hlg':
                out_color_space = to_fn(resized_rgb_image)
                out_color_space_original = to_fn(resized_rgb_image_orignal)
            case 'pq':
                out_color_space = to_fn(resized_rgb_image, div = 100)
                out_color_space_original = to_fn(resized_rgb_image_orignal, div = 100)

        mask_image_1 = Image.open(mask_1_image).convert('RGB')
        mask_image_1_blurred = pipe.mask_processor.blur(mask_image_1, blur_factor=10).convert('RGB')
        mask_image_2 = Image.open(mask_2_image).convert('RGB')
        mask_image_2_blurred = pipe.mask_processor.blur(mask_image_2, blur_factor=10).convert('RGB')
        # mask_image_1_to_save = (np.array(mask_image_1.convert('RGB').copy(), dtype = np.float32) / 255.0).clip(0, 1)
        # save_exr(mask_image_1_to_save, f"{out_sub_folder}/mask_1_sky_blur.exr")
        # mask_image_2_to_save = (np.array(mask_image_2.convert('RGB').copy(), dtype = np.float32) / 255.0).clip(0, 1)
        # save_exr(mask_image_2_to_save, f"{sub_path}/mask_2_light_blur.exr")
        
        normal_output, normal_output_color_space = normal_generation(pipe, out_color_space.copy(), mask_image_1_blurred, mask_image_2_blurred, prompt_1_sky, prompt_2_light, negative_prompt_1_sky, negative_prompt_2_light, args.color_space, from_fn)
        # contrast_output, contrast_output_color_space, scaling_factor = contrast_generation(pipe, out_color_space_original.copy(), mask_image_1.copy(), mask_image_2_blurred.copy(), prompt_1_sky, prompt_2_light, negative_prompt_1_sky, negative_prompt_2_light, args.color_space, from_fn, sub_path)
        # blur_output, blur_output_color_space = blur_generation(pipe, out_color_space_original.copy(), mask_image_1.copy(), mask_image_2_blurred.copy(), prompt_1_sky, prompt_2_light, negative_prompt_1_sky, negative_prompt_2_light, args.color_space, from_fn)
        contrast_output, contrast_output_color_space, scaling_factor = contrast_generation(pipe, out_color_space_original.copy(), mask_image_1.copy(), mask_image_1_blurred.copy(), prompt_1_sky, prompt_2_light, negative_prompt_1_sky, negative_prompt_2_light, args.color_space, from_fn, sub_path)
        blur_output, blur_output_color_space = blur_generation(pipe, out_color_space_original.copy(), mask_image_1.copy(), mask_image_1_blurred.copy(), prompt_1_sky, prompt_2_light, negative_prompt_1_sky, negative_prompt_2_light, args.color_space, from_fn)


        # print(f"{normal_output_color_space.shape}---{normal_output_color_space.min()}---{normal_output_color_space.max()}")

        # Salvataggio immagini
        Image.fromarray((out_color_space * 255.0).astype(np.uint8), mode="RGB").save(input_image_png)
        Image.fromarray((out_color_space_original * 255.0).astype(np.uint8), mode="RGB").save(input_image_original_png)
        # print(f"DEBUG: {out_color_space.shape}---{out_color_space.dtype}")
        # Image.fromarray(out_color_space).save(input_image_png)
        # normal image
        save_exr(normal_output, output_image)
        Image.fromarray(((normal_output_color_space * 255.0).astype(np.uint8)), mode="RGB").save(output_image_png)
        # contrast image
        save_exr(contrast_output, output_image_contrast)
        Image.fromarray(((contrast_output_color_space * 255.0).astype(np.uint8)), mode="RGB").save(output_image_contrast_png)
        # blur image
        save_exr(blur_output, output_image_blur)
        Image.fromarray(((blur_output_color_space * 255.0).astype(np.uint8)), mode="RGB").save(output_image_blur_png)
        mask_image_1_blurred.save(output_mask_1_png)
        mask_image_2_blurred.save(output_mask_2_png)

        # Salvo meta immagine originale e meta generati
        # half_bake = np.zeros(resized_rgb_image.shape, dtype=np.float32)
        # half_bake[0:-1, 0:512, :] = resized_rgb_image[0:-1, 0:512, :]
        # half_bake[0:-1, 512:-1, :] = output_to_save[0:-1, 512:-1, :]
        # save_exr(half_bake, f"{sub_path}/half_img.exr")
        
        # Fill HTML file
        html_file.write(f'<h1>{folder}</h1>')
        html_file.write(f'<p>Input Image</p>')
        html_file.write(f'<img src="./{folder}/input.png" alt="Input">')
        html_file.write(f'<p>Input Image Original (Contrast and Blur)</p>')
        html_file.write(f'<img src="./{folder}/input_original.png" alt="Input Original">')
        html_file.write(f'<p>Mask Image Sky</p>')
        html_file.write(f'<img src="./{folder}/mask_1_sky.png" alt="Mask sky">')
        html_file.write(f'<p>Mask Image Light</p>')
        html_file.write(f'<img src="./{folder}/mask_2_light.png" alt="Mask Light">')
        html_file.write(f'<p>Normal Output Image</p>')
        html_file.write(f'<img src="./{folder}/output_{args.color_space}_{args.model_version}.png" alt="Normal Output">')
        html_file.write(f'<p>Contrast Output Image {scaling_factor}</p>')
        html_file.write(f'<img src="./{folder}/output_contrast_{args.color_space}_{args.model_version}.png" alt="Contrast Output">')
        html_file.write(f'<p>Blur Output Image</p>')
        html_file.write(f'<img src="./{folder}/output_blur_{args.color_space}_{args.model_version}.png" alt="Blur Output">')
        html_file.write("<hr>")
        
        print(f"--------------- END INPAINT {folder} ---------------")
    
    html_file.write("""</body>
</html>""")
    html_file.close()

# python test_inpainting_sdxl.py --model_path ./finetune_lora_sdxl_pq_100/checkpoint-5000 --test_input_dir ../test_input --color_space pq
if __name__ == "__main__":
    main()