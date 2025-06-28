# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia
from typing import Optional, Tuple, Union # Added these imports
from torchvision import transforms

from .affine_transform import AlignRestore
from .face_detector import FaceDetector
from latentsync.utils.util import read_video, write_video # Ensure this import is correct

def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    # Modified to return Optional[Tuple] and handle no face gracefully
    def affine_transform(self, image: Union[torch.Tensor, np.ndarray]) -> Optional[Tuple[torch.Tensor, list, np.ndarray]]:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        
        # Ensure image is in a format the face_detector expects (e.g., numpy HWC, uint8)
        # Assuming face_detector expects numpy array for its input directly
        if isinstance(image, torch.Tensor):
            image_np_for_detector = image.cpu().numpy()
        else:
            image_np_for_detector = image.copy() # Use a copy to avoid side effects

        bbox, landmark_2d_106 = self.face_detector(image_np_for_detector) # Pass numpy array to detector
        
        if bbox is None:
            print("[DEBUG] No face detected in this frame. Returning None from affine_transform.")
            return None # Return None when no face is found

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        # Use the original image_np_for_detector (which is a numpy array) for restorer.align_warp_face
        face_warped_np, affine_matrix = self.restorer.align_warp_face(image_np_for_detector, landmarks3=landmarks3, smooth=True)
        
        box = [0, 0, face_warped_np.shape[1], face_warped_np.shape[0]]  # x1, y1, x2, y2
        
        # Resize to resolution and convert to torch.Tensor CHW for pipeline
        face_final_tensor = cv2.resize(face_warped_np, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face_final_tensor = rearrange(torch.from_numpy(face_final_tensor), "h w c -> c h w")
        
        return face_final_tensor, box, affine_matrix

    # This method is used to prepare inputs for the UNet, specifically for frames WITH faces.
    # It takes an already affine-transformed (aligned) face tensor.
    def preprocess_fixed_mask_image(self, face_tensor: torch.Tensor):
        # face_tensor is expected to be a torch.Tensor in CHW format, already aligned and resized
        pixel_values = self.normalize(face_tensor / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    # This method is designed to take a batch of already-processed face tensors
    # and prepare them for the UNet. It is NOT responsible for handling "no-face" bypass.
    # It assumes `images` are already correctly processed (either actual faces or dummy data).
    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3: # Assuming N H W C, convert to N C H W
            images = rearrange(images, "f h w c -> f c h w")

        # Now `images` should be N C H W torch.Tensor
        
        # Use a list comprehension to apply preprocess_fixed_mask_image
        results = [self.preprocess_fixed_mask_image(image_tensor) for image_tensor in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3: # Assuming N H W C, convert to N C H W
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


# This VideoProcessor class is not directly used by LipsyncPipeline, so no changes needed here.
# It's kept for completeness if other parts of the original repo use it.
class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            # Note: This would also need to handle `None` return if used in a similar fashion.
            result = self.image_processor.affine_transform(frame)
            if result is not None:
                frame, _, _ = result
                results.append(frame)
            # else: do nothing if no face for this processor
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    # Example usage for ImageProcessor if needed for testing
    print("Example usage of ImageProcessor (intended for manual testing/debugging).")
    # This part would typically be used to test components in isolation.
    # It requires dummy video/image files and a 'FaceDetector' setup.# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
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

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia
from typing import Optional, Tuple, Union # Added these imports

from .affine_transform import AlignRestore
from .face_detector import FaceDetector
from latentsync.utils.util import read_video, write_video # Ensure this import is correct

def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    # Modified to return Optional[Tuple] and handle no face gracefully
    def affine_transform(self, image: Union[torch.Tensor, np.ndarray]) -> Optional[Tuple[torch.Tensor, list, np.ndarray]]:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        
        # Ensure image is in a format the face_detector expects (e.g., numpy HWC, uint8)
        # Assuming face_detector expects numpy array for its input directly
        if isinstance(image, torch.Tensor):
            image_np_for_detector = image.cpu().numpy()
        else:
            image_np_for_detector = image.copy() # Use a copy to avoid side effects

        bbox, landmark_2d_106 = self.face_detector(image_np_for_detector) # Pass numpy array to detector
        
        if bbox is None:
            print("[DEBUG] No face detected in this frame. Returning None from affine_transform.")
            return None # Return None when no face is found

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        # Use the original image_np_for_detector (which is a numpy array) for restorer.align_warp_face
        face_warped_np, affine_matrix = self.restorer.align_warp_face(image_np_for_detector, landmarks3=landmarks3, smooth=True)
        
        box = [0, 0, face_warped_np.shape[1], face_warped_np.shape[0]]  # x1, y1, x2, y2
        
        # Resize to resolution and convert to torch.Tensor CHW for pipeline
        face_final_tensor = cv2.resize(face_warped_np, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face_final_tensor = rearrange(torch.from_numpy(face_final_tensor), "h w c -> c h w")
        
        return face_final_tensor, box, affine_matrix

    # This method is used to prepare inputs for the UNet, specifically for frames WITH faces.
    # It takes an already affine-transformed (aligned) face tensor.
    def preprocess_fixed_mask_image(self, face_tensor: torch.Tensor):
        # face_tensor is expected to be a torch.Tensor in CHW format, already aligned and resized
        pixel_values = self.normalize(face_tensor / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    # This method is designed to take a batch of already-processed face tensors
    # and prepare them for the UNet. It is NOT responsible for handling "no-face" bypass.
    # It assumes `images` are already correctly processed (either actual faces or dummy data).
    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3: # Assuming N H W C, convert to N C H W
            images = rearrange(images, "f h w c -> f c h w")

        # Now `images` should be N C H W torch.Tensor
        
        # Use a list comprehension to apply preprocess_fixed_mask_image
        results = [self.preprocess_fixed_mask_image(image_tensor) for image_tensor in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3: # Assuming N H W C, convert to N C H W
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


# This VideoProcessor class is not directly used by LipsyncPipeline, so no changes needed here.
# It's kept for completeness if other parts of the original repo use it.
class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            # Note: This would also need to handle `None` return if used in a similar fashion.
            result = self.image_processor.affine_transform(frame)
            if result is not None:
                frame, _, _ = result
                results.append(frame)
            # else: do nothing if no face for this processor
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    # Example usage for ImageProcessor if needed for testing
    print("Example usage of ImageProcessor (intended for manual testing/debugging).")
    # This part would typically be used to test components in isolation.
    # It requires dummy video/image files and a 'FaceDetector' setup.
