"""
Benchmark and Comparison of OCR Models & Full Pipeline on Validation Set.
Evaluates EasyOCR vs GOT-OCR 2.0 on 251 Indian License Plate Validation Samples.
Metrics:
  - Exact Match (EM) Accuracy (%)
  - Character Accuracy (1 - CER) (%)
  - Indian RTO Grammar Match Rate (%)
  - Average Latency (ms/plate) on RTX 4060 GPU
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline.ocr_engine import EasyOCREngine, GotOCREngine
from anpr_pipeline.grammar_validator import IndianRTOValidator
from anpr_pipeline import ANPRPipeline

def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def compute_character_accuracy(pred: str, gt: str) -> float:
    """Normalized character accuracy (1 - CER). Range: [0.0, 1.0]."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    dist = levenshtein_distance(pred, gt)
    cer = dist / max(len(gt), len(pred))
    return max(0.0, 1.0 - cer)

def evaluate_ocr_engine(engine, engine_name: str, samples: list, validator: IndianRTOValidator):
    print(f"\nEvaluating {engine_name} on {len(samples)} validation samples...")
    em_raw = 0
    em_corrected = 0
    char_accs = []
    latencies = []
    grammar_matches = 0

    results = []

    for idx, (crop_file, gt) in enumerate(samples):
        crop_path = os.path.join("datasets/plate_ocr/crops", crop_file)
        if not os.path.exists(crop_path):
            continue

        img = cv2.imread(crop_path)
        if img is None:
            continue

        # Benchmark latency
        t0 = time.perf_counter()
        raw_pred, conf = engine.read_plate(img)
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(lat)

        # Grammar correction
        corr_pred, gram_score = validator.validate(raw_pred)

        is_em_raw = (raw_pred == gt)
        is_em_corr = (corr_pred == gt)
        c_acc = compute_character_accuracy(corr_pred, gt)

        if is_em_raw:
            em_raw += 1
        if is_em_corr:
            em_corrected += 1
        char_accs.append(c_acc)

        if gram_score >= 0.70:
            grammar_matches += 1

        results.append({
            "image": crop_file,
            "ground_truth": gt,
            "raw_prediction": raw_pred,
            "corrected_prediction": corr_pred,
            "confidence": conf,
            "grammar_score": gram_score,
            "exact_match": is_em_corr,
            "char_accuracy": round(c_acc * 100, 1),
            "latency_ms": round(lat, 1)
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(samples):
            print(f"  [{idx + 1}/{len(samples)}] Progress... Current EM: {em_corrected / (idx + 1) * 100:.1f}%, Avg Latency: {np.mean(latencies):.1f} ms")

    n = len(results)
    summary = {
        "engine": engine_name,
        "total_samples": n,
        "exact_match_raw_pct": round(em_raw / n * 100, 2) if n else 0,
        "exact_match_corrected_pct": round(em_corrected / n * 100, 2) if n else 0,
        "character_accuracy_pct": round(float(np.mean(char_accs)) * 100, 2) if n else 0,
        "grammar_match_pct": round(grammar_matches / n * 100, 2) if n else 0,
        "avg_latency_ms": round(float(np.mean(latencies)), 2) if n else 0,
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2) if n else 0,
        "details": results
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate & Compare ANPR OCR Engines")
    parser.add_argument("--samples", type=int, default=251, help="Number of validation samples to evaluate (default: all 251)")
    parser.add_argument("--save-json", type=str, default="runs/ocr_comparison_results.json", help="Path to save output benchmark JSON")
    args = parser.parse_args()

    # Load validation split
    val_txt = "datasets/plate_ocr/val.txt"
    if not os.path.exists(val_txt):
        print(f"Error: {val_txt} not found.")
        return

    with open(val_txt, "r") as f:
        all_samples = [line.strip().split("\t") for line in f if line.strip()]

    samples = all_samples[:args.samples]
    print("=" * 75)
    print(f"       ANPR OCR MODEL BENCHMARK & COMPARISON (RTX 4060 GPU)")
    print("=" * 75)
    print(f"Dataset:       Indian Vehicle License Plate Dataset (Validation Split)")
    print(f"Total Samples: {len(samples)}")
    print(f"Models:        1) EasyOCR (CRAFT + ResNet-BiLSTM-CTC)")
    print(f"               2) GOT-OCR 2.0 (Stepfun Vision-Language Transformer)")
    print("=" * 75)

    validator = IndianRTOValidator()

    # 1. Evaluate EasyOCR
    print("\n[1/2] Initializing EasyOCR Engine on GPU...")
    easy_engine = EasyOCREngine(gpu=True)
    easy_summary = evaluate_ocr_engine(easy_engine, "EasyOCR", samples, validator)

    # 2. Evaluate GOT-OCR 2.0
    print("\n[2/2] Initializing GOT-OCR 2.0 on GPU...")
    got_engine = GotOCREngine(device="cuda:0")
    got_summary = evaluate_ocr_engine(got_engine, "GOT-OCR 2.0", samples, validator)

    # Comparison Table
    print("\n" + "=" * 75)
    print("                      HEAD-TO-HEAD COMPARISON")
    print("=" * 75)
    header = f"{'Metric':<32} | {'EasyOCR':<18} | {'GOT-OCR 2.0':<18}"
    print(header)
    print("-" * 75)
    print(f"{'Exact Match (Raw OCR)':<32} | {easy_summary['exact_match_raw_pct']:>16.2f}% | {got_summary['exact_match_raw_pct']:>16.2f}%")
    print(f"{'Exact Match (+ RTO Grammar)':<32} | {easy_summary['exact_match_corrected_pct']:>16.2f}% | {got_summary['exact_match_corrected_pct']:>16.2f}%")
    print(f"{'Character Accuracy (1 - CER)':<32} | {easy_summary['character_accuracy_pct']:>16.2f}% | {got_summary['character_accuracy_pct']:>16.2f}%")
    print(f"{'RTO Grammar Syntax Match Rate':<32} | {easy_summary['grammar_match_pct']:>16.2f}% | {got_summary['grammar_match_pct']:>16.2f}%")
    print(f"{'Avg GPU Latency per Plate':<32} | {easy_summary['avg_latency_ms']:>15.1f}ms | {got_summary['avg_latency_ms']:>15.1f}ms")
    print(f"{'95th Percentile Latency':<32} | {easy_summary['p95_latency_ms']:>15.1f}ms | {got_summary['p95_latency_ms']:>15.1f}ms")
    print("=" * 75)

    # Determine Winner
    if got_summary['exact_match_corrected_pct'] > easy_summary['exact_match_corrected_pct']:
        winner = "GOT-OCR 2.0"
        diff = got_summary['exact_match_corrected_pct'] - easy_summary['exact_match_corrected_pct']
    else:
        winner = "EasyOCR"
        diff = easy_summary['exact_match_corrected_pct'] - got_summary['exact_match_corrected_pct']

    print(f"\n[CONCLUSION] Highest Accuracy Model: {winner} (+{diff:.2f}% Exact Match)")
    if easy_summary['avg_latency_ms'] < got_summary['avg_latency_ms']:
        speed_factor = got_summary['avg_latency_ms'] / max(0.1, easy_summary['avg_latency_ms'])
        print(f"[THROUGHPUT] EasyOCR is {speed_factor:.1f}x faster for real-time video streaming.")

    # Save results
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)",
        "total_validation_samples": len(samples),
        "comparison": {
            "easyocr": {k: v for k, v in easy_summary.items() if k != "details"},
            "got_ocr": {k: v for k, v in got_summary.items() if k != "details"}
        },
        "sample_predictions": {
            "easyocr": easy_summary["details"][:10],
            "got_ocr": got_summary["details"][:10]
        }
    }

    with open(args.save_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {args.save_json}")

if __name__ == "__main__":
    main()
