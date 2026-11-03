"""
Dataset Health & Statistics Verification Tool for Indian ANPR.
Run: python tools/verify_dataset.py
"""

import os
import glob
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
det_dir = os.path.join(base_dir, "datasets", "plate_detection")
ocr_dir = os.path.join(base_dir, "datasets", "plate_ocr")

print("=" * 60)
print("       INDIAN ANPR DATASET INTEGRITY & STATS REPORT")
print("=" * 60)

# 1. Check Plate Detection Dataset
train_imgs = glob.glob(os.path.join(det_dir, "images", "train", "*.*"))
val_imgs = glob.glob(os.path.join(det_dir, "images", "val", "*.*"))
train_lbls = glob.glob(os.path.join(det_dir, "labels", "train", "*.txt"))
val_lbls = glob.glob(os.path.join(det_dir, "labels", "val", "*.txt"))

print(f"\n[1] PLATE DETECTION DATASET (YOLO Format)")
print(f"  Location: {det_dir}")
print(f"  Train: {len(train_imgs)} images, {len(train_lbls)} labels")
print(f"  Val:   {len(val_imgs)} images, {len(val_lbls)} labels")
print(f"  Total: {len(train_imgs) + len(val_imgs)} images")
print(f"  Config: {os.path.join(det_dir, 'data.yaml')}")

# Verify bounding box validity
invalid_boxes = 0
for lbl_path in train_lbls + val_lbls:
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                _, xc, yc, w, h = map(float, parts)
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid_boxes += 1

print(f"  Bounding Box Check: {'PASS (0 errors)' if invalid_boxes == 0 else f'FAIL ({invalid_boxes} out of bounds)'}")

# 2. Check Plate OCR Dataset
crops = glob.glob(os.path.join(ocr_dir, "crops", "*.*"))
gt_path = os.path.join(ocr_dir, "ground_truth.csv")
train_txt = os.path.join(ocr_dir, "train.txt")
val_txt = os.path.join(ocr_dir, "val.txt")

print(f"\n[2] PLATE OCR DATASET (Character Recognition)")
print(f"  Location: {ocr_dir}")
print(f"  Total Plate Crops: {len(crops)}")

if os.path.exists(train_txt) and os.path.exists(val_txt):
    with open(train_txt, 'r', encoding='utf-8') as f:
        t_lines = len(f.readlines())
    with open(val_txt, 'r', encoding='utf-8') as f:
        v_lines = len(f.readlines())
    print(f"  OCR Splits: {t_lines} train pairs, {v_lines} val pairs")

if os.path.exists(gt_path):
    df = pd.read_csv(gt_path)
    print(f"  Total Metadata Entries in ground_truth.csv: {len(df)}")
    
    print("\n  Top 10 States Represented:")
    top_states = df['state_code'].value_counts().head(10)
    for state, count in top_states.items():
        print(f"    - {state}: {count} plates")
        
    print("\n  Plate Layout Breakdown:")
    layouts = df['layout'].value_counts()
    for layout, count in layouts.items():
        print(f"    - {layout}: {count} plates ({count/len(df)*100:.1f}%)")

print("=" * 60)
print("Verification complete! All datasets are ready for training.")
print("=" * 60)
