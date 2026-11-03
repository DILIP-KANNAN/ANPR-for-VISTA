"""
License Plate Detector using fine-tuned Ultralytics YOLO.
"""

from typing import Dict, Any, List
import os
import cv2
import numpy as np
import torch
from anpr_pipeline.base import BasePlateDetector

class YOLOPlateDetector(BasePlateDetector):
    def __init__(self, weights_path: str = None, conf_thresh: float = 0.25, device: str = "cuda:0"):
        super().__init__()
        from ultralytics import YOLO

        self.device = device if (torch.cuda.is_available() and "cuda" in device) else "cpu"
        self.conf = conf_thresh

        # Check default trained weights path
        default_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "best_plate.pt")
        if weights_path and os.path.exists(weights_path):
            self.model = YOLO(weights_path)
        elif os.path.exists(default_weights):
            self.model = YOLO(default_weights)
        else:
            # Pretrained fallback
            self.model = YOLO("yolov8n.pt")

    def detect(self, image_or_crop: np.ndarray) -> List[Dict[str, Any]]:
        if image_or_crop is None or image_or_crop.size == 0:
            return []

        h, w = image_or_crop.shape[:2]
        results = self.model(
            source=image_or_crop, 
            conf=self.conf, 
            verbose=False, 
            device=self.device
        )

        detected_plates = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                coords = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = coords
                
                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) > 10 and (y2 - y1) > 8:
                    plate_crop = image_or_crop[y1:y2, x1:x2]
                    detected_plates.append({
                        "plate_bbox": [x1, y1, x2, y2],
                        "plate_crop": plate_crop,
                        "confidence": round(conf, 2)
                    })

        # Sort by confidence descending
        detected_plates.sort(key=lambda x: x["confidence"], reverse=True)
        return detected_plates
