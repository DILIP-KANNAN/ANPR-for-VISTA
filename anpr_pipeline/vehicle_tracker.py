"""
Vehicle Detection and Tracking using Ultralytics YOLO & ByteTrack.
"""

from typing import Dict, Any, List
import numpy as np
from anpr_pipeline.base import BaseVehicleTracker

# COCO class IDs for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class YOLOVehicleTracker(BaseVehicleTracker):
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35, device: str = "cuda:0"):
        super().__init__()
        from ultralytics import YOLO
        import torch
        
        # Select device (cuda if available, else cpu)
        self.device = device if (torch.cuda.is_available() and "cuda" in device) else "cpu"
        self.model = YOLO(model_name)
        self.conf = conf_thresh
        self.track_counter = 1

    def track(self, frame: np.ndarray, timestamp_sec: float) -> List[Dict[str, Any]]:
        """
        Runs tracking across frames. If single image without temporal track ID,
        falls back to detection with deterministic ID assignment.
        """
        try:
            # Run Ultralytics tracker (ByteTrack)
            results = self.model.track(
                source=frame, 
                conf=self.conf, 
                classes=[2, 3, 5, 7], 
                persist=True, 
                verbose=False,
                device=self.device
            )
        except Exception:
            # Fallback to standard detect if tracking state fails
            results = self.model(
                source=frame, 
                conf=self.conf, 
                classes=[2, 3, 5, 7], 
                verbose=False,
                device=self.device
            )

        tracked_vehicles = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                v_type = VEHICLE_CLASS_MAP.get(cls_id, "car")
                coords = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())

                # Get track ID from tracker if present
                if box.id is not None:
                    t_id = int(box.id[0].item())
                else:
                    t_id = self.track_counter
                    self.track_counter += 1

                tracked_vehicles.append({
                    "track_id": t_id,
                    "bbox": coords,
                    "vehicle_type": v_type,
                    "confidence": round(conf, 2)
                })

        # If no vehicle detected (e.g. image is already a tight vehicle crop), use whole frame
        if not tracked_vehicles and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            tracked_vehicles.append({
                "track_id": 1,
                "bbox": [0, 0, w, h],
                "vehicle_type": "car",
                "confidence": 0.90
            })

        return tracked_vehicles
