"""
Lightweight Sub-Pixel Convolutional Neural Network (ESPCN) for License Plate Super-Resolution.
Upscales small and blurred plate crops (< 64px height) by 2x using learned character-edge reconstruction.
Runs natively on GPU with sub-millisecond latency.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn

class PlateESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (Shi et al.)
    3 Conv layers with PixelShuffle(upscale_factor=2).
    """
    def __init__(self, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x))
        return torch.clamp(x, 0.0, 1.0)


class PlateSuperResolver:
    """
    Inference wrapper for Plate Super-Resolution.
    Applies ESPCN to the Luminance (Y) channel in YCrCb color space
    to sharpen character edges without color distortion.
    """
    def __init__(self, weights_path: str = "weights/espcn_plate.pt", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PlateESPCN(upscale_factor=2).to(self.device)
        self.is_loaded = False
        
        if os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(ckpt)
                self.model.eval()
                self.is_loaded = True
            except Exception as e:
                print(f"[PlateSuperResolver] Could not load weights: {e}")

    def upscale_2x(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Upscales an image by 2x using the ESPCN model.
        Falls back to Bicubic interpolation if weights are not available.
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        if not self.is_loaded:
            return cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        try:
            # Convert BGR to YCrCb
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)

            # Prepare Y channel tensor [1, 1, H, W] normalized to [0, 1]
            y_norm = (y.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
            y_tensor = torch.from_numpy(y_norm).to(self.device)

            with torch.no_grad():
                sr_y = self.model(y_tensor)
                sr_y_np = (sr_y.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # Bicubic upscale for chroma channels (human eye is far less sensitive to chroma resolution)
            sr_cr = cv2.resize(cr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            sr_cb = cv2.resize(cb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

            sr_ycrcb = cv2.merge([sr_y_np, sr_cr, sr_cb])
            sr_bgr = cv2.cvtColor(sr_ycrcb, cv2.COLOR_YCrCb2BGR)
            return sr_bgr
        except Exception:
            return cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
