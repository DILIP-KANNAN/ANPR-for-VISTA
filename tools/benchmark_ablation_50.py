"""
Comparative Ablation Benchmark on 50 New Validation Images (ind_0704 to ind_1056).
Compares:
  1. Current Full Pipeline (Plate Enhancer + Character Ambiguity Post-Processing)
  2. Current Pipeline with ONLY Plate Enhancer (No Character Ambiguity)
  3. Previous Pipeline with NO Plate Enhancer & NO Character Ambiguity (Original Baseline)
  4. Previous Pipeline with NO Plate Enhancer & WITH Character Ambiguity
"""

import os
import sys
import re
import glob
import time
import json
import cv2
import numpy as np

# Ensure Windows OpenMP compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline import ANPRPipeline
from anpr_pipeline.base import BaseGrammarValidator
from anpr_pipeline.ocr_engine import get_ocr_engine
from anpr_pipeline.grammar_validator import IndianRTOValidator
from tools.evaluate_pipeline_ocr import compute_character_accuracy
from tools.evaluate_full_pipeline import load_ground_truths

class PassthroughEnhancer:
    """Disables super-resolution, stroke sharpening, and CLAHE; passes raw crop directly."""
    def enhance(self, plate_crop: np.ndarray) -> np.ndarray:
        return plate_crop

class BasicNoAmbiguityValidator(BaseGrammarValidator):
    """Basic alphanumeric normalization without character-level ambiguity resolution."""
    def validate(self, text: str, plate_crop=None):
        if not text:
            return "", 0.0
        clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        score = 0.5 if len(clean) >= 8 else 0.2
        if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$", clean):
            score = 0.99
        return clean, score

def run_benchmark_for_config(name: str, pipeline: ANPRPipeline, images_with_gt: list):
    print(f"\nEvaluating: [{name}] on {len(images_with_gt)} new images...")
    exact_matches = 0
    char_accs = []
    latencies = []
    results = []

    for idx, (img_path, gt) in enumerate(images_with_gt):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        pipeline.reset()
        t0 = time.perf_counter()
        events = pipeline.process_frame(frame)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        if events:
            ev = max(events, key=lambda e: (len(e["anpr"]["plate"]) >= 6, e["anpr"]["grammar_score"], e["anpr"]["confidence"]))
            pred = ev["anpr"]["plate"]
            conf = ev["anpr"]["confidence"]
            gram_score = ev["anpr"]["grammar_score"]
        else:
            pred = ""
            conf = 0.0
            gram_score = 0.0

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
            "latency_ms": round(lat, 1)
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(images_with_gt):
            print(f"  [{idx + 1}/{len(images_with_gt)}] Progress... Current EM: {exact_matches / (idx + 1) * 100:.1f}%, Mean Acc: {np.mean(char_accs) * 100:.1f}%")

    n = len(images_with_gt)
    summary = {
        "config_name": name,
        "sample_count": n,
        "exact_match_rate": round(exact_matches / n * 100, 2),
        "character_accuracy": round(float(np.mean(char_accs)) * 100, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
        "details": results
    }
    return summary

def main():
    print("=" * 85)
    print("      ANPR PIPELINE ABLATION BENCHMARK ON 50 NEW TEST IMAGES")
    print("=" * 85)

    gt_map = load_ground_truths()
    all_val_images = sorted(glob.glob("datasets/plate_detection/images/val/ind_*.jpg"))
    val_images = [img for img in all_val_images if os.path.splitext(os.path.basename(img))[0] in gt_map]
    
    # Select new 50 images (index 100 to 150)
    new_50_images = val_images[100:150]
    images_with_gt = [(img, gt_map[os.path.splitext(os.path.basename(img))[0]]) for img in new_50_images]

    print(f"Total validated images available: {len(val_images)}")
    print(f"Selected slice: Index 100 to 150 ({os.path.basename(new_50_images[0])} to {os.path.basename(new_50_images[-1])})")
    print(f"Evaluating {len(images_with_gt)} new unseen images.")
    print("=" * 85)

    # Initialize shared OCR model once
    print("[GotOCREngine] Initializing shared GOT-OCR 2.0 instance on GPU...")
    shared_ocr = get_ocr_engine("got_ocr")

    # Configuration 1: Current Full Pipeline (Plate Enhancer + Character Ambiguity)
    p1 = ANPRPipeline(
        camera_id="CAM07",
        location="Anna Nagar",
        ocr_engine=shared_ocr,
        plate_weights="weights/best_plate.pt",
        grammar_validator=IndianRTOValidator()
    )

    # Configuration 2: Current Pipeline with ONLY Plate Enhancer (No Character Ambiguity)
    p2 = ANPRPipeline(
        camera_id="CAM07",
        location="Anna Nagar",
        ocr_engine=shared_ocr,
        plate_weights="weights/best_plate.pt",
        grammar_validator=BasicNoAmbiguityValidator()
    )

    # Configuration 3: Previous Pipeline (NO Plate Enhancer & NO Character Ambiguity)
    p3 = ANPRPipeline(
        camera_id="CAM07",
        location="Anna Nagar",
        ocr_engine=shared_ocr,
        plate_weights="weights/best_plate.pt",
        grammar_validator=BasicNoAmbiguityValidator()
    )
    p3.plate_enhancer = PassthroughEnhancer()

    # Configuration 4: Previous Pipeline (NO Plate Enhancer & WITH Character Ambiguity)
    p4 = ANPRPipeline(
        camera_id="CAM07",
        location="Anna Nagar",
        ocr_engine=shared_ocr,
        plate_weights="weights/best_plate.pt",
        grammar_validator=IndianRTOValidator()
    )
    p4.plate_enhancer = PassthroughEnhancer()

    configs = [
        ("Current Pipeline (Enhancer + Ambiguity)", p1),
        ("Current Pipeline (Only Enhancer, No Ambiguity)", p2),
        ("Previous Baseline (No Enhancer, No Ambiguity)", p3),
        ("Previous Pipeline (No Enhancer, With Ambiguity)", p4),
    ]

    all_summaries = {}
    for name, pipe in configs:
        summary = run_benchmark_for_config(name, pipe, images_with_gt)
        all_summaries[name] = summary

    print("\n" + "=" * 90)
    print("                   ABLATION BENCHMARK RESULTS (50 NEW IMAGES)")
    print("=" * 90)
    header = f"{'Pipeline Configuration':<48} | {'Exact Match':<13} | {'Char Accuracy':<14} | {'Latency':<10}"
    print(header)
    print("-" * 90)

    for name in all_summaries:
        s = all_summaries[name]
        em = f"{s['exact_match_rate']:.2f}%"
        ca = f"{s['character_accuracy']:.2f}%"
        lat = f"{s['avg_latency_ms']:.1f} ms"
        print(f"{name:<48} | {em:>13} | {ca:>14} | {lat:>10}")
    print("=" * 90)

    # Save to runs/ablation_benchmark_50.json
    os.makedirs("runs", exist_ok=True)
    out_path = "runs/ablation_benchmark_50.json"
    clean_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)",
        "sample_images": [os.path.basename(img) for img, _ in images_with_gt],
        "metrics": {
            name: {
                "exact_match_rate": all_summaries[name]["exact_match_rate"],
                "character_accuracy": all_summaries[name]["character_accuracy"],
                "avg_latency_ms": all_summaries[name]["avg_latency_ms"]
            }
            for name in all_summaries
        },
        "per_sample_details": {
            name: all_summaries[name]["details"] for name in all_summaries
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, indent=2)

    print(f"\nDetailed ablation results saved to: {out_path}")

if __name__ == "__main__":
    main()
