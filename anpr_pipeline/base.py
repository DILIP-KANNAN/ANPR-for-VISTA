"""
Base interfaces for modular ANPR pipeline components.
Allows hot-swapping any sub-part (detector, tracker, OCR, metadata, fusion).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class BaseVehicleTracker(ABC):
    """Abstract Base Class for Vehicle Detection & Tracking."""
    
    @abstractmethod
    def track(self, frame: np.ndarray, timestamp_sec: float) -> List[Dict[str, Any]]:
        """
        Track vehicles in a video frame or image.
        Returns list of dicts:
        [{
            "track_id": int,
            "bbox": [x1, y1, x2, y2],
            "vehicle_type": str,
            "confidence": float
        }]
        """
        pass

class BaseMetadataEstimator(ABC):
    """Abstract Base Class for Vehicle Metadata (Color, Speed, Direction)."""
    
    @abstractmethod
    def estimate(
        self, 
        frame: np.ndarray, 
        track_id: int, 
        bbox: List[int], 
        vehicle_type: str, 
        timestamp_sec: float
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "type": str,
            "color": str,
            "speed_kmph": float,
            "direction": str
        }
        """
        pass

class BasePlateDetector(ABC):
    """Abstract Base Class for License Plate Detection."""
    
    @abstractmethod
    def detect(self, image_or_crop: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns list of detected plates:
        [{
            "plate_bbox": [x1, y1, x2, y2],
            "plate_crop": np.ndarray,
            "confidence": float
        }]
        """
        pass

class BaseOCREngine(ABC):
    """Abstract Base Class for Character Recognition Engine."""
    
    @abstractmethod
    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Returns: (plate_text, ocr_confidence)
        """
        pass

class BaseGrammarValidator(ABC):
    """Abstract Base Class for Syntax and Grammar Validation."""
    
    @abstractmethod
    def validate(self, text: str) -> Tuple[str, float]:
        """
        Returns: (normalized_plate_text, grammar_score)
        """
        pass

class BaseTemporalFusion(ABC):
    """Abstract Base Class for Multi-frame Tracking & Temporal Fusion."""
    
    @abstractmethod
    def update(
        self, 
        track_id: int, 
        plate_text: str, 
        detection_conf: float, 
        ocr_conf: float, 
        grammar_score: float
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "plate": str,
            "confidence": float,
            "ocr_confidence": float,
            "grammar_score": float,
            "temporal_consistency": float
        }
        """
        pass
