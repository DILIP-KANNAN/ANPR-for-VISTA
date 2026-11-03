"""
Temporal Fusion module for multi-frame ANPR.
Aggregates noisy frame-level OCR readings across a vehicle's tracking trajectory
using confidence-weighted positional voting and computes temporal consistency.
"""

from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
from anpr_pipeline.base import BaseTemporalFusion

def levenshtein_similarity(s1: str, s2: str) -> float:
    """Computes normalized Levenshtein similarity between two strings [0.0 - 1.0]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    return max(0.0, 1.0 - dist / max(m, n))

class TemporalFusion(BaseTemporalFusion):
    def __init__(self, max_history_per_track: int = 30):
        super().__init__()
        self.max_history = max_history_per_track
        # track_id -> list of record dicts
        self.track_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def reset(self, track_id: int = None):
        """Clears track history for a specific track or all tracks."""
        if track_id is not None:
            self.track_history.pop(track_id, None)
        else:
            self.track_history.clear()

    def update(
        self, 
        track_id: int, 
        plate_text: str, 
        detection_conf: float, 
        ocr_conf: float, 
        grammar_score: float
    ) -> Dict[str, Any]:
        """
        Updates the track history and returns fused prediction & temporal consistency.
        """
        record = {
            "text": plate_text,
            "det_conf": max(0.01, detection_conf),
            "ocr_conf": max(0.01, ocr_conf),
            "grammar_score": max(0.01, grammar_score),
            "weight": detection_conf * ocr_conf * (0.5 + 0.5 * grammar_score)
        }

        history = self.track_history[track_id]
        history.append(record)
        if len(history) > self.max_history:
            history.pop(0)

        # 1. Positional confidence voting across all sightings of this track
        # Find the most common plate length
        lengths = [len(r["text"]) for r in history if r["text"]]
        if not lengths:
            target_len = len(plate_text)
        else:
            # Weighted vote on length
            len_votes = defaultdict(float)
            for r in history:
                if r["text"]:
                    len_votes[len(r["text"])] += r["weight"]
            target_len = max(len_votes.items(), key=lambda x: x[1])[0]

        # Character votes per position
        char_votes = defaultdict(lambda: defaultdict(float))
        for r in history:
            txt = r["text"]
            w = r["weight"]
            if len(txt) == target_len:
                for idx, ch in enumerate(txt):
                    char_votes[idx][ch] += w
            elif len(txt) > 0:
                # Approximate alignment
                for idx in range(min(target_len, len(txt))):
                    char_votes[idx][txt[idx]] += w * 0.7

        fused_chars = []
        for idx in range(target_len):
            if idx in char_votes and char_votes[idx]:
                best_char = max(char_votes[idx].items(), key=lambda x: x[1])[0]
                fused_chars.append(best_char)
            elif idx < len(plate_text):
                fused_chars.append(plate_text[idx])

        fused_plate = "".join(fused_chars) if fused_chars else plate_text

        # 2. Compute Temporal Consistency [0.0 - 1.0]
        # Average Levenshtein similarity of history items against the fused plate
        similarities = [levenshtein_similarity(r["text"], fused_plate) for r in history if r["text"]]
        temporal_consistency = float(np.mean(similarities)) if similarities else 0.95

        # Boost consistency if seen across multiple consistent frames
        if len(history) >= 5 and temporal_consistency >= 0.85:
            temporal_consistency = min(0.99, temporal_consistency * 1.05)

        avg_det_conf = float(np.mean([r["det_conf"] for r in history]))
        avg_ocr_conf = float(np.mean([r["ocr_conf"] for r in history]))
        overall_conf = round(avg_det_conf * (0.3 + 0.7 * temporal_consistency), 2)

        return {
            "plate": fused_plate,
            "confidence": min(1.0, max(0.1, overall_conf)),
            "ocr_confidence": round(avg_ocr_conf, 2),
            "grammar_score": round(grammar_score, 2),
            "temporal_consistency": round(temporal_consistency, 2)
        }
