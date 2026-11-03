"""
Evaluation Benchmark for RealtimeEdgeANPRPipeline on 50 and 100 Validation Images.
Compares:
  1. Recognition Accuracy (Exact Match Rate & Character Accuracy)
  2. End-to-End Latency per Frame
  3. Side-by-side comparison with the GOT-OCR Baseline (Option 4)
"""

import os
import sys
import time
import json
import glob
import cv2
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline.realtime_pipeline import RealtimeEdgeANPRPipeline
from tools.evaluate_pipeline_ocr import compute_character_accuracy
from tools.evaluate_full_pipeline import load_ground_truths

def evaluate_realtime_on_images(sample_count: int = 50, enable_tier2: bool = False):
    gt_map = load_ground_truths()
    all_val_images = sorted(glob.glob("datasets/plate_detection/images/val/ind_*.jpg"))
    val_images = [img for img in all_val_images if os.path.splitext(os.path.basename(img))[0] in gt_map][:sample_count]

    print("=" * 80)
    print(f"   EVALUATING REAL-TIME EDGE ANPR PIPELINE ON {len(val_images)} VALIDATION IMAGES")
    print("=" * 80)
    print(f"Sample range: {os.path.basename(val_images[0])} to {os.path.basename(val_images[-1])}")
    print(f"OCR Tier 1: Fast Edge CRNN (weights/crnn_plate.pt) + Beam Search")
    if enable_tier2:
        print(f"OCR Tier 2: GOT-OCR Cascade Fallback Enabled")
    print(f"Post-Processing: Option 4 Character Ambiguity Engine")
    print(f"ReID Engine: OSNet 512-D Embedding & Gallery")
    print("=" * 80)

    tier2_engine = None
    if enable_tier2:
        from anpr_pipeline.ocr_engine import GotOCREngine
        tier2_engine = GotOCREngine()

    pipeline = RealtimeEdgeANPRPipeline(
        camera_id="CAM07",
        location="Anna Nagar",
        enable_reid=True,
        tier2_ocr_engine=tier2_engine
    )

    detected_plates = 0
    exact_matches = 0
    char_accs = []
    latencies = []
    results = []

    for idx, img_path in enumerate(val_images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt = gt_map.get(stem, "")
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        pipeline.reset()
        t0 = time.perf_counter()
        events = pipeline.process_frame(frame)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        if events:
            # Pick best vehicle event with plate
            best_ev = max(events, key=lambda e: (len(e["anpr"]["plate"]) >= 6, e["anpr"]["grammar_score"], e["anpr"]["confidence"]))
            pred = best_ev["anpr"]["plate"]
            conf = best_ev["anpr"]["confidence"]
            gram_score = best_ev["anpr"]["grammar_score"]
            reid_id = best_ev["vehicle"].get("reid", {}).get("matched_vehicle_id", "")
        else:
            pred = ""
            conf = 0.0
            gram_score = 0.0
            reid_id = ""

        if pred:
            detected_plates += 1

        is_em = (pred == gt)
        c_acc = compute_character_accuracy(pred, gt)
        if is_em:
            exact_matches += 1
        char_accs.append(c_acc)

        results.append({
            "image": os.path.basename(img_path),
            "ground_truth": gt,
            "prediction": pred,
            "exact_match": is_em,
            "char_accuracy": round(c_acc, 4),
            "confidence": conf,
            "grammar_score": gram_score,
            "reid_id": reid_id,
            "latency_ms": round(lat, 1)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(val_images):
            print(f"  [{idx + 1:03d}/{len(val_images)}] Latency: {np.mean(latencies):.1f} ms | Exact Match: {exact_matches / (idx + 1) * 100:.1f}% | Char Acc: {np.mean(char_accs) * 100:.1f}%")

    n = len(val_images)
    summary = {
        "sample_count": n,
        "plate_detection_rate": round(detected_plates / n * 100, 2),
        "exact_match_rate": round(exact_matches / n * 100, 2),
        "character_accuracy": round(float(np.mean(char_accs)) * 100, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(latencies)), 1),
        "details": results
    }

    print("\n" + "=" * 80)
    print(f"              BENCHMARK RESULTS ({n} IMAGES)")
    print("=" * 80)
    print(f"Plate Detection Rate             : {summary['plate_detection_rate']:.2f}%")
    print(f"Exact Match Accuracy             : {summary['exact_match_rate']:.2f}%")
    print(f"Character Recognition Accuracy   : {summary['character_accuracy']:.2f}%")
    print(f"Average End-to-End Latency       : {summary['avg_latency_ms']:.1f} ms")
    print(f"Single-Image Throughput          : {summary['throughput_fps']:.1f} FPS")
    print("=" * 80)

    # Save report
    out_file = f"runs/realtime_eval_{n}_images.json"
    os.makedirs("runs", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Report saved to: {out_file}\n")

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50, choices=[50, 100], help="Number of images to evaluate (50 or 100)")
    parser.add_argument("--tier2", action="store_true", help="Enable Tier-2 GOT-OCR hierarchical cascade fallback")
    args = parser.parse_args()
    evaluate_realtime_on_images(args.samples, enable_tier2=args.tier2)

if __name__ == "__main__":
    main()

