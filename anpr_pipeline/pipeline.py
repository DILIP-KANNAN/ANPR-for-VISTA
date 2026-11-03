"""
Complete End-to-End Modular ANPR Pipeline.
Integrates Vehicle Tracking, Metadata Estimation, Plate Detection, OCR,
Grammar Validation, and Temporal Fusion.
Outputs the exact required JSON event schema.
"""

from typing import Dict, Any, List, Optional
import datetime
import cv2
import numpy as np

from anpr_pipeline.base import (
    BaseVehicleTracker, 
    BaseMetadataEstimator, 
    BasePlateDetector, 
    BaseOCREngine, 
    BaseGrammarValidator, 
    BaseTemporalFusion
)
from anpr_pipeline.vehicle_tracker import YOLOVehicleTracker
from anpr_pipeline.metadata_estimator import VehicleMetadataEstimator
from anpr_pipeline.plate_detector import YOLOPlateDetector
from anpr_pipeline.ocr_engine import get_ocr_engine
from anpr_pipeline.grammar_validator import IndianRTOValidator
from anpr_pipeline.temporal_fusion import TemporalFusion
from anpr_pipeline.plate_enhancer import PlateEnhancer

class ANPRPipeline:
    def __init__(
        self,
        camera_id: str = "CAM07",
        location: str = "Anna Nagar",
        ocr_engine_type: str = "easyocr",
        plate_weights: str = None,
        tracker: Optional[BaseVehicleTracker] = None,
        metadata_estimator: Optional[BaseMetadataEstimator] = None,
        plate_detector: Optional[BasePlateDetector] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        grammar_validator: Optional[BaseGrammarValidator] = None,
        temporal_fusion: Optional[BaseTemporalFusion] = None
    ):
        self.camera_id = camera_id
        self.location = location
        self.event_counter = 124

        # Modular components with dependency injection
        self.tracker = tracker or YOLOVehicleTracker()
        self.metadata_estimator = metadata_estimator or VehicleMetadataEstimator()
        self.plate_detector = plate_detector or YOLOPlateDetector(weights_path=plate_weights)
        self.ocr_engine = ocr_engine or get_ocr_engine(ocr_engine_type)
        self.grammar_validator = grammar_validator or IndianRTOValidator()
        self.temporal_fusion = temporal_fusion or TemporalFusion()
        self.plate_enhancer = PlateEnhancer()

    def reset(self):
        """Clears all tracking, trajectory, and temporal fusion history."""
        self.temporal_fusion.reset()
        self.metadata_estimator.reset()
        if hasattr(self.tracker, "track_counter"):
            self.tracker.track_counter = 1

    def process_frame(
        self, 
        frame: np.ndarray, 
        timestamp_str: str = None,
        timestamp_sec: float = None
    ) -> List[Dict[str, Any]]:
        """
        Processes a video frame or image and returns a list of ANPR event dictionaries.
        """
        now = datetime.datetime.now()
        t_str = timestamp_str or now.strftime("%Y-%m-%dT%H:%M:%S")
        t_sec = timestamp_sec or now.timestamp()

        # Step 1: Detect & Track Vehicles
        vehicles = self.tracker.track(frame, t_sec)

        events = []
        for v in vehicles:
            track_id = v["track_id"]
            v_bbox = v["bbox"]
            v_type = v["vehicle_type"]

            # Step 2: Estimate Vehicle Metadata (Type, Color, Speed, Direction)
            metadata = self.metadata_estimator.estimate(
                frame=frame,
                track_id=track_id,
                bbox=v_bbox,
                vehicle_type=v_type,
                timestamp_sec=t_sec
            )

            # Crop vehicle area for plate detector
            vx1, vy1, vx2, vy2 = v_bbox
            vh, vw = frame.shape[:2]
            vehicle_crop = frame[max(0, vy1):min(vh, vy2), max(0, vx1):min(vw, vx2)]
            if vehicle_crop.size == 0:
                vehicle_crop = frame

            # Step 3: Plate Detection (Full Frame + Vehicle Crop matching to avoid bounding box clipping)
            full_plates = self.plate_detector.detect(frame)
            matched_plate = None
            if full_plates:
                # Find the plate inside this vehicle's bounding box
                for fp in full_plates:
                    fx1, fy1, fx2, fy2 = fp["plate_bbox"]
                    fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    if vx1 <= fcx <= vx2 and vy1 <= fcy <= vy2:
                        matched_plate = fp
                        break
                if matched_plate is None and len(vehicles) == 1:
                    matched_plate = full_plates[0]

            if matched_plate is None:
                crop_plates = self.plate_detector.detect(vehicle_crop)
                if crop_plates:
                    matched_plate = crop_plates[0]

            best_plate_crop = matched_plate["plate_crop"] if matched_plate else vehicle_crop
            det_conf = matched_plate["confidence"] if matched_plate else 0.85

            # Step 3.5: Plate Dewarping, Aspect Normalization & Contrast Enhancement
            enhanced_plate = self.plate_enhancer.enhance(best_plate_crop)

            # Step 4: OCR Engine
            raw_text, ocr_conf = self.ocr_engine.read_plate(enhanced_plate)

            # Step 5: Grammar & Syntax Validation with Visual Ambiguity Resolution
            norm_text, grammar_score = self.grammar_validator.validate(raw_text, plate_crop=enhanced_plate)

            # Step 6: Temporal Fusion
            fused_anpr = self.temporal_fusion.update(
                track_id=track_id,
                plate_text=norm_text,
                detection_conf=det_conf,
                ocr_conf=ocr_conf,
                grammar_score=grammar_score
            )

            # Step 7: Construct Final Event Dict matching target schema
            event = {
                "event_id": f"evt_{self.event_counter:05d}",
                "camera_id": self.camera_id,
                "timestamp": t_str,
                "location": self.location,
                "vehicle": {
                    "track_id": metadata["track_id"],
                    "type": metadata["type"],
                    "color": metadata["color"],
                    "speed_kmph": metadata["speed_kmph"],
                    "direction": metadata["direction"]
                },
                "anpr": {
                    "plate": fused_anpr["plate"],
                    "confidence": fused_anpr["confidence"],
                    "ocr_confidence": fused_anpr["ocr_confidence"],
                    "grammar_score": fused_anpr["grammar_score"],
                    "temporal_consistency": fused_anpr["temporal_consistency"]
                }
            }

            self.event_counter += 1
            events.append(event)

        return events
