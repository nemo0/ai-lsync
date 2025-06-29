# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms

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
from ..utils.image_processor import ImageProcessor, load_fixed_mask # load_fixed_mask is still external
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

    # THIS STATIC METHOD IS CORRECTLY LOCATED AND MODIFIED HERE
    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # `masks` tensor (passed as 3rd arg) is 0.0 for lips (black in your PNG), 1.0 for face (white in your PNG).
        # We want to apply `decoded_latents` where the mask is black (lips).
        # We want to preserve `pixel_values` where the mask is white (face).

        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype) 
        
        # CORRECTED LOGIC:
        combined_pixel_values = decoded_latents * (1 - masks) + pixel_values * masks 

        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray) -> (List[Optional[torch.Tensor]], List[Optional[list]], List[Optional[np.ndarray]], List[bool]):
        """
        Processes video frames to detect faces. If no face is detected in a frame,
        it appends None for face, box, and affine_matrix, and False to has_face_map.
        Returns:
            - faces: List of preprocessed face tensors (or None)
            - boxes: List of bounding boxes (or None)
            - affine_matrices: List of affine transformation matrices (or None)
            - has_face_map: Boolean list, True if face detected, False otherwise
        """
        faces = []
        boxes = []
        affine_matrices = []
        has_face_map = [] # True if face detected, False otherwise
        
        print(f"Detecting faces in {len(video_frames)} frames...")
        for i, frame in enumerate(tqdm.tqdm(video_frames)):
            try:
                face, box, affine_matrix = self.image_processor.affine_transform(frame)
                faces.append(face) # face is already a torch.Tensor here
                boxes.append(box)
                affine_matrices.append(affine_matrix)
                has_face_map.append(True)
            except RuntimeError as e:
                if "Face not detected" in str(e):
                    faces.append(None)
                    boxes.append(None)
                    affine_matrices.append(None)
                    has_face_map.append(False)
                else:
                    raise e # Re-raise any other unexpected RuntimeErrors
        return faces, boxes, affine_matrices, has_face_map

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        """
        This is the original restore_video function. It should only be called with
        faces that have successfully been lipsynced (i.e., had faces detected and processed).
        `faces` here is a batch of (C, resolution, resolution) lipsynced cropped faces.
        `video_frames` here is a batch of corresponding original full frames.
        `boxes` and `affine_matrices` are corresponding lists of original detected face data.
        """
        # Ensure input lengths match - crucial for correct mapping
        if faces.shape[0] != video_frames.shape[0] or \
           faces.shape[0] != len(boxes) or \
           faces.shape[0] != len(affine_matrices):
            print("Warning: Mismatch in input lengths for restore_video. Proceeding with min length.")
            min_len = min(faces.shape[0], video_frames.shape[0], len(boxes), len(affine_matrices))
            faces = faces[:min_len]
            video_frames = video_frames[:min_len]
            boxes = boxes[:min_len]
            affine_matrices = affine_matrices[:min_len]

        out_frames = []
        for index, lipsynced_cropped_face_tensor in enumerate(tqdm.tqdm(faces, desc="Restoring lipsynced faces...")): # Renamed `face` to clarify content
            x1, y1, x2, y2 = boxes[index]
            height_orig_face = int(y2 - y1)
            width_orig_face = int(x2 - x1)
            
            # This step resizes the (C, resolution, resolution) lipsynced cropped face
            # to the actual bounding box dimensions (C, height_orig_face, width_orig_face).
            processed_face_tensor_for_restore_img = torchvision.transforms.functional.resize(
                lipsynced_cropped_face_tensor, 
                size=(height_orig_face, width_orig_face), 
                interpolation=transforms.InterpolationMode.BICUBIC, 
                antialias=True
            )
            
            # self.image_processor.restorer.restore_img blends the processed face region
            # back onto the original full frame using the affine matrix.
            out_frame = self.image_processor.restorer.restore_img(
                video_frames[index], # Original full frame for this index
                processed_face_tensor_for_restore_img, # The resized lipsynced face region
                affine_matrices[index]              # Affine matrix for this face
            )
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # This function remains unchanged from your original.
        # It handles looping/trimming video frames to match audio length.
        if len(whisper_chunks) > len(video_frames):
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                else:
                    loop_video_frames.append(video_frames[::-1])
            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
        return video_frames # Return the effective video frames

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16, # This is the batch size for UNet, not number of frames to process
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask3.png", # Updated default mask path
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
        # The mask_image is loaded here and passed to ImageProcessor
        mask_image = load_fixed_mask(height, mask_image_path) # Call global function
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Processing video frames...")

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

        # 5. Read audio and video
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        original_video_frames_raw = read_video(video_path, use_decord=False) 

        # Get the effective video frames after looping/trimming based on audio length
        effective_video_frames = self.loop_video(whisper_chunks, original_video_frames_raw)
        
        # Initialize the list for final output frames with copies of the effective_video_frames.
        # Frames will be replaced in this list ONLY IF a face is detected and processed.
        final_synced_frames_list = [f for f in effective_video_frames] # Make a mutable list copy

        # Perform face detection for all effective video frames.
        # This will return three lists for faces, boxes, and affine matrices,
        # where elements are None if no face was found, and also a boolean map.
        all_detected_faces, all_detected_boxes, all_detected_affine_matrices, has_face_map = \
            self.affine_transform_video(effective_video_frames)

        num_channels_latents = self.vae.config.latent_channels
        
        # Collect data for frames that actually need processing (i.e., have faces)
        # Store tuples of (original_idx, face_tensor, box, affine_matrix, audio_chunk)
        data_for_unet_processing = []
        for i in range(len(effective_video_frames)):
            if has_face_map[i]:
                data_for_unet_processing.append({
                    "original_idx": i,
                    "face": all_detected_faces[i],
                    "box": all_detected_boxes[i],
                    "affine_matrix": all_detected_affine_matrices[i],
                    "audio_chunk": whisper_chunks[i] # Ensure audio chunk aligns
                })
        
        if not data_for_unet_processing: # Check if the list is empty (no faces detected at all)
            print("No faces detected in any frame. Returning original video.")
            # Skip all UNet processing, proceed to video/audio writing
            synced_video_frames_np = np.stack(final_synced_frames_list, axis=0) # Convert back to numpy array
        else:
            # Iterate through data_for_unet_processing in batches of `num_frames`
            num_batches = math.ceil(len(data_for_unet_processing) / num_frames)
            
            print(f"Total batches with faces for UNet inference: {num_batches}")

            for batch_num in range(num_batches):
                start_idx_batch = batch_num * num_frames
                end_idx_batch = min((batch_num + 1) * num_frames, len(data_for_unet_processing))
                current_batch_data = data_for_unet_processing[start_idx_batch:end_idx_batch]
                
                batch_size_current = len(current_batch_data) # Actual size of this specific batch

                # Collect data for the current batch
                batch_original_indices = [item["original_idx"] for item in current_batch_data]
                batch_input_faces = torch.stack([item["face"] for item in current_batch_data]) # Stack faces for UNet input
                batch_audio_embeds = torch.stack([item["audio_chunk"] for item in current_batch_data])
                batch_boxes = [item["box"] for item in current_batch_data]
                batch_affine_matrices = [item["affine_matrix"] for item in current_batch_data]
                
                # --- START ORIGINAL UNET PROCESSING FLOW FOR A BATCH (Restored) ---
                # This section should be as close as possible to how it worked before "no-face" handling.

                # Prepare latents for the current batch
                batch_latents = self.prepare_latents(
                    batch_size_current, # Use actual batch size
                    num_channels_latents,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                )

                # Audio embeds for UNet
                audio_embeds_for_unet = batch_audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds_for_unet)
                    audio_embeds_for_unet = torch.cat([null_audio_embeds, audio_embeds_for_unet])
                
                # Mask and Masked Image Latent preparation
                # `masks` here will be 0 for lips, 1 for face.
                ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                    batch_input_faces, affine_transform=False # batch_input_faces is list of tensors
                )
                mask_latents, masked_image_latents = self.prepare_mask_latents(
                    masks, masked_pixel_values, height, width, weight_dtype, device, generator, do_classifier_free_guidance
                )

                # Reference Image Latent preparation
                ref_latents = self.prepare_image_latents(
                    ref_pixel_values, device, weight_dtype, generator, do_classifier_free_guidance
                )

                # Denoising loop
                current_latents = batch_latents
                num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
                with self.progress_bar(total=num_inference_steps) as progress_bar: 
                    for j, t in enumerate(timesteps):
                        unet_input = torch.cat([current_latents] * 2) if do_classifier_free_guidance else current_latents
                        unet_input = self.scheduler.scale_model_input(unet_input, t)
                        unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
                        
                        noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds_for_unet).sample
                        
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                            
                        current_latents = self.scheduler.step(noise_pred, t, current_latents, **extra_step_kwargs).prev_sample
                        
                        if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                            if callback is not None and j % callback_steps == 0:
                                callback(j, t, current_latents)

                # Recover the pixel values for the current batch of generated faces
                decoded_latents = self.decode_latents(current_latents)
                
                # Apply the staticmethod `paste_surrounding_pixels_back` from THIS class.
                # It now correctly handles your mask convention (0=lips, 1=face)
                # This returns the lipsynced cropped face (C, resolution, resolution)
                generated_lipsynced_cropped_faces_batch = self.paste_surrounding_pixels_back(
                    decoded_latents, ref_pixel_values, masks, device, weight_dtype # Pass original `masks` (0=lips, 1=face)
                )
                
                # --- END ORIGINAL UNET PROCESSING FLOW FOR A BATCH ---

                # Now, use the original `restore_video` method to blend these processed faces
                # back onto their respective original full frames.
                
                # Gather original full frames for this batch
                batch_original_full_frames = np.stack([effective_video_frames[idx] for idx in batch_original_indices], axis=0)

                # Call the original restore_video function
                restored_full_frames_for_batch = self.restore_video(
                    generated_lipsynced_cropped_faces_batch, # Faces from UNet (C, resolution, resolution)
                    batch_original_full_frames,              # Corresponding original full frames (N, H, W, C)
                    batch_boxes,                             # Corresponding original boxes
                    batch_affine_matrices                    # Corresponding original affine matrices
                )
                
                # Update the final list with the restored full frames
                for k, original_idx in enumerate(batch_original_indices):
                    final_synced_frames_list[original_idx] = restored_full_frames_for_batch[k]


            # Convert list of frames back to numpy array for video writing
            synced_video_frames_np = np.stack(final_synced_frames_list, axis=0)

        # Audio processing (remains unchanged as it's continuous)
        audio_samples = read_audio(audio_path)
        audio_samples_remain_length = int(synced_video_frames_np.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames_np, fps=video_fps)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)