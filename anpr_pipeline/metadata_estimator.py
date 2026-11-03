"""
Vehicle Metadata Estimator (Type, Color, Speed, Direction).
"""

from collections import defaultdict
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import math
from anpr_pipeline.base import BaseMetadataEstimator

# Standard vehicle colors in HSV range
COLOR_RANGES = [
    ("white",  (0, 0, 180),    (180, 40, 255)),
    ("black",  (0, 0, 0),      (180, 255, 55)),
    ("silver", (0, 0, 100),    (180, 35, 180)),
    ("red",    (0, 70, 50),    (10, 255, 255)),
    ("red",    (170, 70, 50),  (180, 255, 255)),
    ("yellow", (15, 60, 60),   (35, 255, 255)),
    ("blue",   (95, 60, 50),   (130, 255, 255)),
    ("green",  (35, 60, 50),   (85, 255, 255)),
    ("grey",   (0, 0, 55),     (180, 30, 110))
]

def angle_to_compass(degrees: float) -> str:
    """Maps an angle in degrees [0, 360) to compass direction."""
    compass_points = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    idx = int(round(degrees / 45.0)) % 8
    return compass_points[idx]

class VehicleMetadataEstimator(BaseMetadataEstimator):
    def __init__(self, meters_per_pixel: float = 0.05):
        super().__init__()
        self.m_per_px = meters_per_pixel
        # track_id -> list of (x_center, y_center, timestamp_sec)
        self.trajectories: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        self.colors: Dict[int, str] = {}

    def reset(self, track_id: int = None):
        """Clears trajectory and color history."""
        if track_id is not None:
            self.trajectories.pop(track_id, None)
            self.colors.pop(track_id, None)
        else:
            self.trajectories.clear()
            self.colors.clear()

    def estimate_color(self, vehicle_crop: np.ndarray) -> str:
        """Estimates dominant color of the vehicle body."""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return "unknown"

        h, w = vehicle_crop.shape[:2]
        # Crop central body area to avoid windshield glass and tires/road
        y1, y2 = int(h * 0.25), int(h * 0.80)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        body = vehicle_crop[y1:y2, x1:x2]
        if body.size == 0:
            body = vehicle_crop

        hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
        
        # Count pixels matching each color range
        color_counts = defaultdict(int)
        for name, lower, upper in COLOR_RANGES:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            color_counts[name] += cv2.countNonZero(mask)

        if not color_counts:
            return "white"

        best_color = max(color_counts.items(), key=lambda x: x[1])[0]
        return best_color

    def estimate_motion(
        self, 
        track_id: int, 
        bbox: List[int], 
        timestamp_sec: float
    ) -> Tuple[float, str]:
        """
        Estimates speed in km/h and direction heading from centroid trajectory.
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        history = self.trajectories[track_id]
        history.append((cx, cy, timestamp_sec))
        if len(history) > 20:
            history.pop(0)

        if len(history) < 2:
            # Default initial baseline
            return 40.0, "N"

        # Compare oldest in window with newest
        old_x, old_y, old_t = history[0]
        new_x, new_y, new_t = history[-1]
        dt = max(0.001, new_t - old_t)

        dx = new_x - old_x
        dy = new_y - old_y

        # Distance in pixels -> meters -> speed km/h
        pixel_dist = math.hypot(dx, dy)
        meters = pixel_dist * self.m_per_px
        speed_mps = meters / dt
        speed_kmph = speed_mps * 3.6

        # Cap speed to realistic traffic range
        speed_kmph = max(5.0, min(120.0, speed_kmph))

        # Direction calculation (dy is positive downwards in screen space)
        # In screen space: -dy is North, +dy is South, +dx is East, -dx is West
        angle_rad = math.atan2(-dy, dx)
        angle_deg = (math.degrees(angle_rad) + 360) % 360
        direction = angle_to_compass(angle_deg)

        return round(speed_kmph, 1), direction

    def estimate(
        self, 
        frame: np.ndarray, 
        track_id: int, 
        bbox: List[int], 
        vehicle_type: str, 
        timestamp_sec: float
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        if track_id in self.colors:
            color = self.colors[track_id]
        else:
            color = self.estimate_color(crop)
            self.colors[track_id] = color

        speed_kmph, direction = self.estimate_motion(track_id, bbox, timestamp_sec)

        return {
            "track_id": track_id,
            "type": vehicle_type.lower() if vehicle_type else "car",
            "color": color,
            "speed_kmph": speed_kmph,
            "direction": direction
        }
