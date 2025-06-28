# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia


class AlignRestore(object):
    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float16):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = resolution / 256 * 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.device = device
            self.dtype = dtype
            # Fill value should ideally be in normalized range if used with normalized inputs/outputs
            self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype) / 255.0 
            self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

    def align_warp_face(self, img, landmarks3, smooth=True):
        affine_matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )

        # img input is uint8 (0-255)
        # Convert img to float 0-1 range CHW before kornia
        img_tensor_norm = rearrange(torch.from_numpy(img).float() / 255.0, "h w c -> c h w").unsqueeze(0)
        
        # --- FIX: Move img_tensor_norm to the correct device and dtype ---
        img_tensor_norm = img_tensor_norm.to(device=self.device, dtype=self.dtype)
        # --- END FIX ---

        affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        cropped_face = kornia.geometry.transform.warp_affine(
            img_tensor_norm, # Now this tensor is on the GPU
            affine_matrix,   # This tensor is also on the GPU
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        # Output from kornia is 0-1 float. Convert back to 0-255 uint8 for numpy.
        cropped_face = rearrange(cropped_face.squeeze(0) * 255.0, "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        h, w, _ = input_img.shape

        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        # --- FIX FOR `face` input ---
        # 'face' is a NumPy array (HWC, 0-255 uint8) from pixel_values_to_images.
        # Convert it to a PyTorch tensor (CHW, 0-1 float) before moving to device and unsqueezing.
        face_tensor_norm = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0 # CHW, float32, 0-1 range
        face_to_kornia = face_tensor_norm.to(device=self.device, dtype=self.dtype).unsqueeze(0) # CHW, 0-1 float, with batch dim
        # --- END FIX FOR `face` input ---

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        
        inv_mask = kornia.geometry.transform.warp_affine(
            self.mask, inv_affine_matrix, (h, w), padding_mode="zeros"
        ) # inv_mask is now (1, 1, H_out, W_out)
        
        inv_face = kornia.geometry.transform.warp_affine(
            face_to_kornia, # Use the properly normalized and typed face tensor
            inv_affine_matrix, 
            (h, w), 
            mode="bilinear", 
            padding_mode="fill", 
            fill_value=self.fill_value # This fill_value is 0-1
        ).squeeze(0) # inv_face is now CHW, 0-1 float (squeeze batch dim for expand_as later)

        # --- FIX FOR WHITISH OVERLAY: REMOVE INCORRECT DENORMALIZATION ---
        # inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255 # <--- REMOVE THIS LINE ENTIRELY
        # --- END FIX ---

        input_img_tensor_norm = rearrange(torch.from_numpy(input_img).float() / 255.0, "h w c -> c h w")
        input_img_tensor_norm = input_img_tensor_norm.to(device=self.device, dtype=self.dtype)

        inv_mask_erosion = kornia.morphology.erosion(
            inv_mask, # This is now 4D: (1, 1, H, W)
            torch.ones(
                (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
            ),
        ) # 0-1 float, (1, 1, H, W)

        # --- FIX: MOVE DEFINITION OF w_edge AND erosion_radius HERE ---
        # These need to be defined before they are used in cv2.erode
        total_face_area = torch.sum(inv_mask_erosion.float())
        w_edge = int(total_face_area**0.5) // 20
        erosion_radius = w_edge * 2
        # --- END FIX ---

        inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face) # inv_mask_erosion is (1,1,H,W), squeeze(0) makes it (1,H,W) for expand_as(CHW)
        pasted_face = inv_mask_erosion_t * inv_face # (0-1 float) * (0-1 float) = 0-1 float (generated face region)
        
        inv_mask_center_np = cv2.erode(inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32), np.ones((erosion_radius, erosion_radius), np.uint8))
        inv_mask_center_tensor = torch.from_numpy(inv_mask_center_np).to(device=self.device, dtype=self.dtype)[None, None, ...] # Re-add 2 singleton dims

        blur_size = w_edge * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        inv_soft_mask = kornia.filters.gaussian_blur2d(
            inv_mask_center_tensor, (blur_size, blur_size), (sigma, sigma)
        ).squeeze(0) # 0-1 float (soft blending mask). Squeeze batch dim for expand_as later.
        inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)

        # Final blend: all tensors are now in the 0-1 float range
        img_back_norm = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img_tensor_norm

        # Convert final blended image back to HWC 0-255 uint8 for NumPy output
        img_back = rearrange(img_back_norm * 255.0, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = img_back.cpu().numpy()
        return img_back

    def transformation_from_points(self, points1: torch.Tensor, points0: torch.Tensor, smooth=True, p_bias=None):
        if isinstance(points0, np.ndarray):
            points2 = torch.tensor(points0, device=self.device, dtype=torch.float32)
        else:
            points2 = points0.clone()

        if isinstance(points1, np.ndarray):
            points1_tensor = torch.tensor(points1, device=self.device, dtype=torch.float32)
        else:
            points1_tensor = points1.clone()

        c1 = torch.mean(points1_tensor, dim=0)
        c2 = torch.mean(points2, dim=0)

        points1_centered = points1_tensor - c1
        points2_centered = points2 - c2

        s1 = torch.std(points1_centered)
        s2 = torch.std(points2_centered)

        points1_normalized = points1_centered / s1
        points2_normalized = points2_centered / s2

        covariance = torch.matmul(points1_normalized.T, points2_normalized)
        U, S, V = torch.svd(covariance.float())

        R = torch.matmul(V, U.T)

        det = torch.det(R.float())
        if det < 0:
            V[:, -1] = -V[:, -1]
            R = torch.matmul(V, U.T)

        sR = (s2 / s1) * R
        T = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(R, c1.reshape(2, 1))

        M = torch.cat((sR, T), dim=1)

        if smooth:
            bias = points2_normalized[2] - points1_normalized[2]
            if p_bias is None:
                p_bias = bias
            else:
                bias = p_bias * 0.2 + bias * 0.8
            p_bias = bias
            M[:, 2] = M[:, 2] + bias

        return M.cpu().numpy(), p_bias
