# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union, Tuple
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms # Ensure this is imported

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

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
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, original_full_frame_pixels, masks, device, weight_dtype):
        # This function is typically used for pixel-level blending within a fixed canvas.
        # It's intended input is a generated region and a background.
        # For full face restoration to original video resolution, `restore_img` is used.
        # This function might be called internally by the UNet to process inputs.
        original_full_frame_pixels = original_full_frame_pixels.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + original_full_frame_pixels * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    # REVISED: This method now returns a list of (face_tensor, box, affine_matrix) or None for each frame
    def affine_transform_video(self, video_frames_input: np.ndarray) -> List[Optional[Tuple[torch.Tensor, list, np.ndarray]]]:
        all_per_frame_processed_data = [] # Will store (face_tensor, box, affine_matrix) or None for each frame
        
        print(f"Affine transforming {len(video_frames_input)} faces...")
        for frame_idx, frame_np in enumerate(tqdm.tqdm(video_frames_input)):
            # Call the image_processor's affine_transform method
            result = self.image_processor.affine_transform(frame_np)

            if result is None: # No face detected
                all_per_frame_processed_data.append(None)
            else: # Face detected
                face, box, affine_matrix = result
                all_per_frame_processed_data.append((face, box, affine_matrix))
                
        return all_per_frame_processed_data

    # This method's logic is now integrated into __call__ directly.
    # It's kept here just in case other parts of the original code rely on it.
    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    # REVISED: loop_video now returns looped original frames and the per-frame processed data
    def loop_video(self, whisper_chunks: list, video_frames_original_full: np.ndarray) -> Tuple[np.ndarray, List[Optional[Tuple[torch.Tensor, list, np.ndarray]]]]:
        # Determine the target number of frames based on audio length
        num_target_frames = len(whisper_chunks)

        # If the audio is longer than the video, we need to loop the video frames
        if num_target_frames > len(video_frames_original_full):
            num_loops = math.ceil(num_target_frames / len(video_frames_original_full))
            
            looped_original_video_frames_list = []
            for i in range(num_loops):
                if i % 2 == 0:
                    looped_original_video_frames_list.append(video_frames_original_full)
                else:
                    # Reverse video frames for odd loops to make it seamless
                    looped_original_video_frames_list.append(video_frames_original_full[::-1])
            
            # Concatenate and trim to the exact target length
            final_original_frames_np = np.concatenate(looped_original_video_frames_list, axis=0)[:num_target_frames]
        else:
            # If video is long enough, just trim it
            final_original_frames_np = video_frames_original_full[:num_target_frames]
        
        # Now, process all these final_original_frames_np to get face data or None
        # This will call the (modified) self.affine_transform_video
        all_per_frame_processed_data = self.affine_transform_video(final_original_frames_np)

        return final_original_frames_np, all_per_frame_processed_data

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16, # This is the batch size for UNet
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 5. Audio and Video Preparation
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        original_video_frames_initial = read_video(video_path, use_decord=False) # Read video once

        # Loop video frames and get processed data or None for each frame
        # `final_original_frames_np` will be the looped/trimmed original video frames (NumPy array)
        # `all_per_frame_processed_data` will be a list where each element is (face_tensor,box,mat) or None
        final_original_frames_np, all_per_frame_processed_data = self.loop_video(whisper_chunks, original_video_frames_initial)

        # This list will hold the final video frames as NumPy arrays, ready for `write_video`
        final_output_frames_list = [None] * len(final_original_frames_np) 
        
        num_channels_latents = self.vae.config.latent_channels
        
        # 6. Main Inference Loop - Process in chunks for UNet efficiency
        num_total_frames = len(final_original_frames_np)
        
        for i in tqdm.tqdm(range(0, num_total_frames, num_frames), desc="Processing video chunks..."):
            current_chunk_original_frames = final_original_frames_np[i : i + num_frames]
            current_chunk_processed_data = all_per_frame_processed_data[i : i + num_frames]
            current_chunk_whisper_chunks = whisper_chunks[i : i + num_frames] # Audio embeds for this chunk

            # Lists to build the batch for UNet inference
            unet_batch_ref_pixel_values = []        # For UNet input (cropped face)
            unet_batch_masked_pixel_values = []
            unet_batch_masks = []
            unet_batch_audio_embeds = []
            
            # This maps the index *within the UNet batch* to its absolute index in `final_output_frames_list`
            unet_batch_to_final_idx_map = [] 

            for j, frame_data in enumerate(current_chunk_processed_data):
                absolute_frame_idx = i + j
                original_frame_for_current_pos = current_chunk_original_frames[j] # NumPy HWC, original pixels

                if frame_data is None: # No face detected for this frame
                    # This frame bypasses UNet. We put the original resized frame directly into the final list.
                    if original_frame_for_current_pos.shape[0] != height or original_frame_for_current_pos.shape[1] != width:
                        resized_original_frame = cv2.resize(original_frame_for_current_pos, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        resized_original_frame = original_frame_for_current_pos
                    
                    final_output_frames_list[absolute_frame_idx] = resized_original_frame
                else: # Face detected - prepare for UNet processing
                    face_tensor, box_list, affine_matrix_np = frame_data # face_tensor is torch.Tensor, CHW (cropped face)
                    
                    # Prepare inputs for UNet (normalize, apply mask)
                    face_tensor_normalized = self.image_processor.normalize(face_tensor / 255.0)
                    masked_face_tensor = face_tensor_normalized * self.image_processor.mask_image # Apply the fixed mask
                    
                    # Add data to UNet batch lists
                    unet_batch_ref_pixel_values.append(face_tensor_normalized) # Cropped face for UNet input
                    unet_batch_masked_pixel_values.append(masked_face_tensor)
                    unet_batch_masks.append(self.image_processor.mask_image[0:1]) # The single-channel mask
                    unet_batch_audio_embeds.append(current_chunk_whisper_chunks[j]) # Audio embeds for this specific frame
                    
                    # Add original frame's box and affine matrix to lists for restoration
                    # This is implicitly retrieved from frame_data during `affine_transform_video`
                    # but we need to store them explicitly if they are not part of `frame_data` itself.
                    # As per previous agreement, `frame_data` already contains (face, box, affine_matrix)
                    
                    unet_batch_to_final_idx_map.append(absolute_frame_idx)

            # --- Run UNet Inference for the current batch of FACES ONLY ---
            if unet_batch_ref_pixel_values: # Only proceed if there are frames with faces in this chunk
                ref_pixel_values_batch_tensor = torch.stack(unet_batch_ref_pixel_values).to(device, dtype=weight_dtype)
                masked_pixel_values_batch_tensor = torch.stack(unet_batch_masked_pixel_values).to(device, dtype=weight_dtype)
                masks_batch_tensor = torch.stack(unet_batch_masks).to(device, dtype=weight_dtype)
                audio_embeds_batch_tensor = torch.stack(unet_batch_audio_embeds).to(device, dtype=weight_dtype)
                
                # Apply classifier-free guidance for audio embeds
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds_batch_tensor)
                    audio_embeds_batch_tensor = torch.cat([null_audio_embeds, audio_embeds_batch_tensor])

                # Prepare latents for *only* the frames that will go through UNet
                latents_for_unet_batch = self.prepare_latents(
                    len(unet_batch_ref_pixel_values), # Actual number of frames in this UNet batch
                    num_channels_latents,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                )
                
                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks_batch_tensor, masked_pixel_values_batch_tensor, height, width, weight_dtype, device, generator, do_classifier_free_guidance
                )
                ref_latents = self.prepare_image_latents(
                    ref_pixel_values_batch_tensor, device, weight_dtype, generator, do_classifier_free_guidance
                )

                # Denoising loop for the current UNet batch
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for j, t in enumerate(timesteps):
                        unet_input = torch.cat([latents_for_unet_batch] * 2) if do_classifier_free_guidance else latents_for_unet_batch
                        unet_input = self.scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds_batch_tensor).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                        latents_for_unet_batch = self.scheduler.step(noise_pred, t, latents_for_unet_batch, **extra_step_kwargs).prev_sample
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, latents_for_unet_batch)

                # Recover the pixel values for the processed batch (this is the GENERATED FACE)
                decoded_latents_batch = self.decode_latents(latents_for_unet_batch)
                
                # Convert generated face tensor to NumPy (0-255, HWC)
                generated_face_np_batch = self.pixel_values_to_images(decoded_latents_batch) # N_batch, H, W, C
                
                # --- REINTRODUCE RESTORE_IMG FOR EACH PROCESSED FRAME ---
                # Loop through the processed items in this UNet batch
                for k, final_idx_in_output_list in enumerate(unet_batch_to_final_idx_map):
                    # Get the original full-resolution frame for this specific position
                    # Note: final_original_frames_np is already the looped/trimmed version
                    original_full_frame_k_np = final_original_frames_np[final_idx_in_output_list] # NumPy HWC, original resolution

                    # Get the generated face (512x512 NumPy) for this specific item in the UNet batch
                    generated_face_k_np = generated_face_np_batch[k] # NumPy HWC, 512x512

                    # Retrieve the original box and affine matrix for this specific frame
                    # from the 'all_per_frame_processed_data' list
                    # This relies on all_per_frame_processed_data having the same order as final_original_frames_np
                    # We know frame_data was (face, box, affine_matrix)
                    _, box_k, affine_matrix_k = all_per_frame_processed_data[final_idx_in_output_list] 
                    
                    # Perform restoration using the restorer (AlignRestore.restore_img)
                    # This puts the generated face back into the original full frame
                    restored_frame_np = self.image_processor.restorer.restore_img(
                        original_full_frame_k_np, # Original full resolution frame (NumPy)
                        generated_face_k_np,      # Generated face (512x512 NumPy)
                        affine_matrix_k           # Affine matrix to put it back in place
                    )
                    
                    # --- FIX FOR ALL INPUT ARRAYS MUST HAVE SAME SHAPE: RESIZE THE RESTORED FRAME TO TARGET HEIGHT/WIDTH ---
                    # Ensure all frames inserted into final_output_frames_list have the exact same dimensions.
                    # This is your target model resolution (e.g., 512x512)
                    if restored_frame_np.shape[0] != height or restored_frame_np.shape[1] != width:
                        final_frame_to_add = cv2.resize(restored_frame_np, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        final_frame_to_add = restored_frame_np
                    # --- END FIX ---

                    final_output_frames_list[final_idx_in_output_list] = final_frame_to_add
        
        # After the entire loop, `final_output_frames_list` should be fully populated
        # Stack all collected NumPy frames into a single array
        final_video_output_np = np.stack(final_output_frames_list, axis=0)

        # --- Debugging prints for final video content ---
        print(f"[DEBUG] Final video output NP - Shape: {final_video_output_np.shape}")
        print(f"[DEBUG] Final video output NP - Dtype: {final_video_output_np.dtype}")
        print(f"[DEBUG] Final video output NP - Number of frames: {final_video_output_np.shape[0] if final_video_output_np.ndim > 0 else 0}")
        
        if final_video_output_np.size > 0 and np.all(final_video_output_np == 0):
            print("[DEBUG] WARNING: final_video_output_np contains all zero pixel values!")
        
        if final_video_output_np.shape[0] > 0 and final_video_output_np.ndim == 4: # N, H, W, C
            print(f"[DEBUG] Final video output NP - First 5 frames, top-left pixel (R): {final_video_output_np[:5, 0, 0, 0]}")
        # --- End Debugging prints ---


        audio_samples_remain_length = int(final_video_output_np.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        # Ensure temp_dir exists for intermediate files (though it should already from modal_app.py)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        # Write intermediate video and audio files
        write_video(os.path.join(temp_dir, "video.mp4"), final_video_output_np, fps=video_fps)
        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        # FFmpeg command to combine video and audio
        # Note: If output quality is still an issue, try increasing num_inference_steps (e.g., to 50 or 100)
        # in the main call (or FastAPI parameter). A higher CRF would make quality *worse*.
        command = f"ffmpeg -y -loglevel info -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        
        # Capture FFmpeg output for better debugging
        print(f"[DEBUG] Running FFmpeg command: {command}")
        result_ffmpeg = subprocess.run(command, shell=True, capture_output=True, text=True) 
        
        print(f"[DEBUG] FFmpeg return code: {result_ffmpeg.returncode}")
        if result_ffmpeg.stdout:
            print(f"[DEBUG] FFmpeg stdout:\n{result_ffmpeg.stdout}")
        if result_ffmpeg.stderr:
            print(f"[DEBUG] FFmpeg stderr:\n{result_ffmpeg.stderr}")
        
        if result_ffmpeg.returncode != 0:
            print(f"[ERROR] FFmpeg command failed with return code {result_ffmpeg.returncode}")
            raise RuntimeError(f"FFmpeg command failed. Stderr: {result_ffmpeg.stderr}")

        print(f"[DEBUG] FFmpeg command completed. Final video expected at: {video_out_path}")
