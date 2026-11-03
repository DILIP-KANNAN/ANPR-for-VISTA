"""
End-to-End Pipeline Evaluation on the Validation Set.
Evaluates:
  1. Vehicle Tracking & Detection
  2. License Plate Detection (YOLO best_plate.pt)
  3. OCR Recognition (EasyOCR vs GOT-OCR 2.0)
  4. Grammar Validation & Output Event Generation
"""

import os
import sys

# Prevent Windows libiomp5 duplicate initialization clash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import time
import json
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline import ANPRPipeline
from tools.evaluate_pipeline_ocr import compute_character_accuracy

def load_ground_truths():
    """Maps validation image base name to ground truth plate text."""
    gt_map = {}
    csv_path = "datasets/plate_ocr/ground_truth.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 8 and parts[7] == "val":
                    crop_file = parts[0]
                    plate_text = parts[1]
                    # Map crop file back to source image
                    # e.g., crop_0001_0_MH47Y1124.jpg -> ind_0001
                    # dc_extra_...
                    if crop_file.startswith("crop_"):
                        # crop_0001_0_... -> ind_0001
                        num_str = crop_file.split("_")[1]
                        gt_map[f"ind_{num_str}"] = plate_text
                    elif "dc_extra" in crop_file:
                        name_stem = "_".join(crop_file.split("_")[:3])
                        gt_map[name_stem] = plate_text
    return gt_map

def run_pipeline_eval(ocr_engine_type: str, max_samples: int = 50):
    gt_map = load_ground_truths()
    all_val_images = sorted(glob.glob("datasets/plate_detection/images/val/ind_*.jpg"))
    val_images = [img for img in all_val_images if os.path.splitext(os.path.basename(img))[0] in gt_map]
    if max_samples:
        val_images = val_images[:max_samples]

    print(f"\nEvaluating Full Pipeline with OCR = '{ocr_engine_type}' on {len(val_images)} validated images...")
    pipeline = ANPRPipeline(
        camera_id="CAM07",
        location="Anna Nagar",
        ocr_engine_type=ocr_engine_type,
        plate_weights="weights/best_plate.pt"
    )

    detected_plates = 0
    exact_matches = 0
    char_accuracies = []
    latencies = []
    valid_grammar = 0

    results = []

    for i, img_path in enumerate(val_images):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        expected_gt = gt_map.get(stem, "")

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Reset pipeline state so previous image tracks don't bleed into next image
        pipeline.reset()

        t0 = time.perf_counter()
        events = pipeline.process_frame(frame)
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(lat)

        if events:
            ev = max(events, key=lambda e: (len(e["anpr"]["plate"]) >= 6, e["anpr"]["grammar_score"], e["anpr"]["confidence"]))
            pred_plate = ev["anpr"]["plate"]
            conf = ev["anpr"]["confidence"]
            gram_score = ev["anpr"]["grammar_score"]

            if pred_plate:
                detected_plates += 1

            is_em = False
            c_acc = 0.0
            if expected_gt:
                is_em = (pred_plate == expected_gt)
                c_acc = compute_character_accuracy(pred_plate, expected_gt)
                char_accuracies.append(c_acc)
                if is_em:
                    exact_matches += 1

            if gram_score >= 0.70:
                valid_grammar += 1

            results.append({
                "file": os.path.basename(img_path),
                "ground_truth": expected_gt,
                "predicted_plate": pred_plate,
                "confidence": conf,
                "grammar_score": gram_score,
                "exact_match": is_em,
                "latency_ms": round(lat, 1)
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(val_images):
            print(f"  [{i+1}/{len(val_images)}] Processed... Latency: {np.mean(latencies):.1f}ms")

    n = len(val_images)
    n_gt = len(char_accuracies)
    summary = {
        "engine": ocr_engine_type,
        "total_images": n,
        "plate_detection_rate": round(detected_plates / n * 100, 2),
        "exact_match_rate": round(exact_matches / n_gt * 100, 2) if n_gt else 0.0,
        "char_accuracy": round(float(np.mean(char_accuracies)) * 100, 2) if n_gt else 0.0,
        "valid_grammar_rate": round(valid_grammar / n * 100, 2),
        "avg_pipeline_latency_ms": round(float(np.mean(latencies)), 2),
        "details": results
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Number of images to evaluate (default: 50)")
    parser.add_argument("--ocr", type=str, default="all", choices=["all", "easyocr", "got_ocr", "crnn", "ensemble"], help="OCR engine to evaluate")
    parser.add_argument("--save-json", type=str, default="runs/full_pipeline_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 75)
    print("           END-TO-END ANPR PIPELINE VALIDATION BENCHMARK")
    print("=" * 75)
    print("Stages: Tracking -> Plate Detection (YOLO) -> Plate Enhancer -> OCR -> RTO")
    print(f"Sample Size: {args.samples} validation images | Engine(s): {args.ocr}")
    print("=" * 75)

    engines = ["easyocr", "got_ocr", "ensemble"] if args.ocr == "all" else [args.ocr]
    eval_results = {}

    for eng in engines:
        res = run_pipeline_eval(eng, max_samples=args.samples)
        eval_results[eng] = res

    print("\n" + "=" * 85)
    print("                      FULL PIPELINE ACCURACY & PERFORMANCE")
    print("=" * 85)
    header = f"{'Metric':<32}"
    for eng in engines:
        header += f" | {eng.upper():<14}"
    print(header)
    print("-" * 85)

    for metric_key, metric_label in [
        ("plate_detection_rate", "Plate Detection Rate"),
        ("exact_match_rate", "End-to-End Exact Match"),
        ("char_accuracy", "Character Recognition Acc"),
        ("valid_grammar_rate", "RTO Grammar Compliance"),
        ("avg_pipeline_latency_ms", "Avg Latency per Frame")
    ]:
        row = f"{metric_label:<32}"
        for eng in engines:
            val = eval_results[eng][metric_key]
            if "latency" in metric_key:
                row += f" | {val:>11.1f} ms"
            else:
                row += f" | {val:>13.2f}%"
        print(row)
    print("=" * 85)

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)",
        "samples_evaluated": args.samples,
        "results": {eng: {k: v for k, v in eval_results[eng].items() if k != "details"} for eng in engines}
    }
    with open(args.save_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark saved to: {args.save_json}")

if __name__ == "__main__":
    main()
