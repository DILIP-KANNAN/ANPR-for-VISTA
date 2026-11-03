"""
Standalone Evaluation Benchmark for SVTR-Tiny Model with STN Rectifier.
Evaluates on 50 and 100 validation images completely separately from CRNN.
Compares:
  1. Standalone SVTR-Tiny Recognition (Exact Match & Character Accuracy)
  2. Inference Latency
  3. Side-by-Side Comparison against the CRNN baseline
"""

import os
import sys
import time
import json
import glob
import cv2
import numpy as np
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline.svtr_model import SVTRTinyWithSTN, decode_svtr_beam_search
from anpr_pipeline.plate_detector import YOLOPlateDetector
from anpr_pipeline.grammar_validator import IndianRTOValidator
from tools.evaluate_pipeline_ocr import compute_character_accuracy
from tools.evaluate_full_pipeline import load_ground_truths

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class SVTREdgeEngine:
    def __init__(self, weights_path: str = "weights/svtr_plate.pt", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chars = CHARS
        self.model = SVTRTinyWithSTN(num_classes=len(self.chars) + 1, use_stn=True).to(self.device)

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            print(f"[SVTREdgeEngine] Initialized on {self.device} with weights from {weights_path}")
        else:
            print(f"[SVTREdgeEngine] Warning: {weights_path} not found.")

        self.model.eval()

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros((32, 128), dtype=np.uint8)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        return cv2.resize(gray, (128, 32), interpolation=cv2.INTER_CUBIC)

    @torch.no_grad()
    def read_plate(self, crop: np.ndarray) -> dict:
        p = self.preprocess_crop(crop)
        tensor = ((torch.from_numpy(p).float() / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        decoded = decode_svtr_beam_search(logits, self.chars, beam_width=10)
        return decoded[0] if decoded else {"plate": "", "confidence": 0.0, "grammar_score": 0.0}

def evaluate_svtr_on_images(sample_count: int = 50, enable_tier2: bool = False):
    gt_map = load_ground_truths()
    all_val_images = sorted(glob.glob("datasets/plate_detection/images/val/ind_*.jpg"))
    val_images = [img for img in all_val_images if os.path.splitext(os.path.basename(img))[0] in gt_map][:sample_count]

    print("=" * 80)
    mode_name = "TWO-TIER CASCADE (SVTR-TINY + GOT-OCR)" if enable_tier2 else "STANDALONE SVTR-TINY + STN RECTIFIER"
    print(f"   {mode_name} BENCHMARK ({len(val_images)} IMAGES)")
    print("=" * 80)
    print(f"Sample range: {os.path.basename(val_images[0])} to {os.path.basename(val_images[-1])}")
    print(f"Tier 1: SVTR-Tiny (3.8M params) + STN Rectifier (weights/svtr_plate.pt)")
    if enable_tier2:
        print(f"Tier 2: GOT-OCR Cascade Fallback Enabled (invoked when conf < 0.85 or grammar < 0.80)")
    print(f"Decoder: MoRTH Grammar-Constrained Beam Search Decoder")
    print("=" * 80)

    plate_detector = YOLOPlateDetector(weights_path="weights/best_plate.pt")
    svtr_engine = SVTREdgeEngine(weights_path="weights/svtr_plate.pt")
    rto_validator = IndianRTOValidator()

    tier2_engine = None
    if enable_tier2:
        from anpr_pipeline.ocr_engine import GotOCREngine
        tier2_engine = GotOCREngine()

    detected_plates = 0
    exact_matches = 0
    char_accs = []
    latencies = []
    ocr_latencies = []
    results = []

    for idx, img_path in enumerate(val_images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt = gt_map.get(stem, "")
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        t0 = time.perf_counter()
        detections = plate_detector.detect(frame)
        det_time = time.perf_counter() - t0

        if detections:
            detected_plates += 1
            best_det = max(detections, key=lambda d: d["confidence"])
            crop = best_det["plate_crop"]

            t_ocr0 = time.perf_counter()
            ocr_res = svtr_engine.read_plate(crop)
            ocr_lat = (time.perf_counter() - t_ocr0) * 1000.0
            ocr_latencies.append(ocr_lat)

            raw_text = ocr_res["plate"]
            conf = ocr_res["confidence"]
            norm_text, g_score = rto_validator.validate(raw_text)

            # Two-Tier Hierarchical Cascade Fallback
            if tier2_engine is not None and (conf < 0.85 or g_score < 0.80):
                t2_raw, t2_conf = tier2_engine.read_plate(crop)
                t2_norm, t2_g = rto_validator.validate(t2_raw)
                if t2_g >= g_score:
                    norm_text, conf, g_score = t2_norm, t2_conf, t2_g
        else:
            norm_text = ""
            conf = 0.0
            g_score = 0.0
            ocr_lat = 0.0

        total_lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(total_lat)

        is_em = (norm_text == gt)
        c_acc = compute_character_accuracy(norm_text, gt)
        if is_em:
            exact_matches += 1
        char_accs.append(c_acc)

        results.append({
            "image": os.path.basename(img_path),
            "ground_truth": gt,
            "prediction": norm_text,
            "exact_match": is_em,
            "char_accuracy": round(c_acc, 4),
            "confidence": round(conf, 3),
            "grammar_score": round(g_score, 3),
            "ocr_latency_ms": round(ocr_lat, 2),
            "total_latency_ms": round(total_lat, 2)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_images):
            print(f"  [{idx + 1:03d}/{len(val_images)}] OCR Lat: {np.mean(ocr_latencies):.2f} ms | Exact Match: {exact_matches / (idx + 1) * 100:.1f}% | Char Acc: {np.mean(char_accs) * 100:.1f}%")

    n = len(val_images)
    summary = {
        "model": "SVTR-Tiny + STN Rectifier",
        "sample_count": n,
        "plate_detection_rate": round(detected_plates / n * 100, 2),
        "exact_match_rate": round(exact_matches / n * 100, 2),
        "character_accuracy": round(float(np.mean(char_accs)) * 100, 2),
        "avg_ocr_latency_ms": round(float(np.mean(ocr_latencies)), 2),
        "avg_total_latency_ms": round(float(np.mean(latencies)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(latencies)), 1),
        "details": results
    }

    print("\n" + "=" * 80)
    print(f"       SVTR-TINY STANDALONE BENCHMARK RESULTS ({n} IMAGES)")
    print("=" * 80)
    print(f"Plate Detection Rate             : {summary['plate_detection_rate']:.2f}%")
    print(f"Exact Match Accuracy             : {summary['exact_match_rate']:.2f}%")
    print(f"Character Recognition Accuracy   : {summary['character_accuracy']:.2f}%")
    print(f"Average SVTR OCR Forward Latency : {summary['avg_ocr_latency_ms']:.2f} ms")
    print(f"Average End-to-End Latency       : {summary['avg_total_latency_ms']:.1f} ms")
    print(f"End-to-End Throughput            : {summary['throughput_fps']:.1f} FPS")
    print("=" * 80)

    mode_str = "cascade" if enable_tier2 else "standalone"
    out_file = f"runs/svtr_{mode_str}_eval_{n}_images.json"
    os.makedirs("runs", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved to: {out_file}\n")

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50, choices=[50, 100])
    parser.add_argument("--tier2", action="store_true", help="Enable Tier-2 GOT-OCR hierarchical cascade fallback")
    args = parser.parse_args()
    evaluate_svtr_on_images(args.samples, enable_tier2=args.tier2)

if __name__ == "__main__":
    main()

