"""
Plate Enhancement & Perspective Rectification Module.
Standardizes plate crops before OCR:
  1. Horizontal Dewarping / Deskewing via Minimum Area Bounding Rect / Hough Angle
  2. Aspect Ratio Normalization & Bicubic Upscaling
  3. Bilateral Edge-Preserving Denoising & CLAHE Contrast Boosting
"""

import cv2
import numpy as np
from anpr_pipeline.super_resolution import PlateSuperResolver

class PlateEnhancer:
    def __init__(self, target_height: int = 96, target_width: int = 384, sr_weights: str = "weights/espcn_plate.pt"):
        self.target_height = target_height
        self.target_width = target_width
        self.sr = PlateSuperResolver(weights_path=sr_weights)

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Corrects small rotation angles (slanted plates) via minAreaRect."""
        if image is None or image.size == 0:
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        pts = np.column_stack(np.where(edges > 0))
        
        if len(pts) < 50:
            return image

        rect = cv2.minAreaRect(pts)
        angle = rect[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        if abs(angle) > 1.0 and abs(angle) < 15.0:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image

    def enhance(self, plate_crop: np.ndarray) -> np.ndarray:
        """Applies deskewing, adaptive super-resolution, edge-preserving sharpening, and CLAHE."""
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # 1. Deskew
        straight = self.deskew(plate_crop)

        # 2. Deep Sub-Pixel Super-Resolution for severely low-res plates (< 32px height)
        h, w = straight.shape[:2]
        if h < 32 and self.sr.is_loaded:
            straight = self.sr.upscale_2x(straight)
            h, w = straight.shape[:2]

        # 3. Aspect Ratio & Dimension Normalization
        if h < self.target_height:
            scale = self.target_height / float(h)
            new_w = max(int(w * scale), self.target_width // 2)
            straight = cv2.resize(straight, (new_w, self.target_height), interpolation=cv2.INTER_CUBIC)

        # 4. High-Frequency Character Stroke Sharpening (Unsharp Mask)
        # Accentuates character strokes (e.g. distinguishing 5 vs 9, E vs M, 0 vs D)
        smooth = cv2.GaussianBlur(straight, (0, 0), 2.5)
        sharpened = cv2.addWeighted(straight, 1.4, smooth, -0.4, 0)

        # 5. Bilateral Edge-Preserving Denoising
        denoised = cv2.bilateralFilter(sharpened, d=5, sigmaColor=50, sigmaSpace=50)

        # 6. CLAHE Contrast Normalization
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

