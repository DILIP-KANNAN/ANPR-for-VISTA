"""
CLI Runner for Modular ANPR Pipeline.
Runs the complete detection, metadata, OCR, grammar, and temporal fusion pipeline
and prints the target JSON event structure.

Usage:
    python run_pipeline.py --input datasets/plate_detection/images/val/ind_0001.jpg
    python run_pipeline.py --ocr got_ocr
"""

import os
import json
import argparse
import glob
import cv2
from anpr_pipeline import ANPRPipeline

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Modular ANPR Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to input image or video file")
    parser.add_argument("--camera-id", type=str, default="CAM07", help="Camera identifier")
    parser.add_argument("--location", type=str, default="Anna Nagar", help="Location name")
    parser.add_argument("--ocr", type=str, default="easyocr", choices=["easyocr", "got_ocr"], help="OCR Engine type")
    parser.add_argument("--weights", type=str, default="weights/best_plate.pt", help="Path to trained plate weights")
    parser.add_argument("--save-json", type=str, default=None, help="Path to save output JSON event(s)")
    parser.add_argument("--draw", type=str, default=None, help="Path to save annotated visual output image")
    parser.add_argument("--benchmark", type=int, default=None, help="Run automated evaluation on N validation images")
    args = parser.parse_args()

    # Benchmark mode over validation set
    if args.benchmark is not None:
        val_images = sorted(glob.glob("datasets/plate_detection/images/val/*.jpg"))[:args.benchmark]
        if not val_images:
            print("[Error] No validation images found for benchmark.")
            return

        print(f"=== Running ANPR Benchmark on {len(val_images)} validation images (OCR: {args.ocr}) ===")
        pipeline = ANPRPipeline(
            camera_id=args.camera_id,
            location=args.location,
            ocr_engine_type=args.ocr,
            plate_weights=args.weights if os.path.exists(args.weights) else None
        )

        detected_count = 0
        valid_grammar_count = 0
        all_events = []

        for i, img_path in enumerate(val_images):
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            evts = pipeline.process_frame(frame)
            if evts:
                best_evt = evts[0]
                all_events.append(best_evt)
                plate = best_evt["anpr"]["plate"]
                score = best_evt["anpr"]["grammar_score"]
                conf = best_evt["anpr"]["confidence"]
                if plate:
                    detected_count += 1
                if score >= 0.7:
                    valid_grammar_count += 1
                print(f"[{i+1}/{len(val_images)}] {os.path.basename(img_path)} -> Plate: {plate:<12} | Conf: {conf:.2f} | Grammar: {score:.2f}")

        print("\n" + "=" * 60)
        print("                 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total Evaluated:        {len(val_images)}")
        print(f"Plates Recognized:      {detected_count}/{len(val_images)} ({detected_count/len(val_images)*100:.1f}%)")
        print(f"High Grammar Matches:   {valid_grammar_count}/{len(val_images)} ({valid_grammar_count/len(val_images)*100:.1f}%)")
        print("=" * 60)

        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(all_events, f, indent=2)
            print(f"Saved benchmark events to: {args.save_json}")
        return

    # If no input provided, pick the first validation image
    if not args.input:
        val_cands = glob.glob("datasets/plate_detection/images/val/*.jpg")
        if val_cands:
            args.input = val_cands[0]
        else:
            print("No test input provided and no validation images found.")
            return

    print(f"Initializing ANPR Pipeline (Camera: {args.camera_id}, Location: {args.location}, OCR: {args.ocr})...")
    pipeline = ANPRPipeline(
        camera_id=args.camera_id,
        location=args.location,
        ocr_engine_type=args.ocr,
        plate_weights=args.weights if os.path.exists(args.weights) else None
    )

    print(f"\nProcessing input: {args.input}")
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Failed to load image: {args.input}")
        return

    # Process frame
    events = pipeline.process_frame(frame)

    print("\n" + "=" * 60)
    print("                    ANPR EVENT OUTPUT")
    print("=" * 60)
    for event in events:
        print(json.dumps(event, indent=2))
    print("=" * 60)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(events, f, indent=2)
        print(f"\nSaved event JSON to: {args.save_json}")

    if args.draw and events:
        vis_frame = frame.copy()
        for ev in events:
            v_info = ev["vehicle"]
            a_info = ev["anpr"]
            label = f"{v_info['color'].title()} {v_info['type'].title()} | {a_info['plate']} ({a_info['confidence']:.2f})"
            cv2.putText(vis_frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            meta_label = f"Speed: {v_info['speed_kmph']} km/h | Dir: {v_info['direction']} | Camera: {ev['camera_id']}"
            cv2.putText(vis_frame, meta_label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.imwrite(args.draw, vis_frame)
        print(f"Saved annotated visual to: {args.draw}")

if __name__ == "__main__":
    main()
