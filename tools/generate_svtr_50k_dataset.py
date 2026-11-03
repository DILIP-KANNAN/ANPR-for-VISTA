"""
Dedicated Synthetic Indian License Plate Generator for SVTR-Tiny (50,000 samples).
Generates diverse 1-line and 2-line Indian HSRP plates across all 38 State/UT codes,
with 4 plate color categories, perspective distortion, blur, noise, and bolt occlusions.
Outputs to datasets/plate_ocr/synthetic_50k/ and datasets/plate_ocr/train_svtr_50k.txt
"""

import os
import random
import string
import argparse
import concurrent.futures
import cv2
import numpy as np

INDIAN_STATES = [
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", 
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", 
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", 
    "WB", "BH", "OR"
]

VALID_SERIES_LETTERS = [c for c in string.ascii_uppercase if c not in ("I", "O")]

def random_plate_number():
    state = random.choice(INDIAN_STATES)
    rto = f"{random.randint(1, 99):02d}"
    plate_type = random.random()
    if plate_type < 0.70:
        series = "".join(random.choices(VALID_SERIES_LETTERS, k=2))
        digits = f"{random.randint(1, 9999):04d}"
    elif plate_type < 0.90:
        series = "".join(random.choices(VALID_SERIES_LETTERS, k=1))
        digits = f"{random.randint(1, 9999):04d}"
    else:
        series = ""
        digits = f"{random.randint(1, 9999):04d}"
    return f"{state}{rto}{series}{digits}"

def generate_single_plate(idx: int, output_dir: str):
    plate_text = random_plate_number()
    layout = "1-line" if random.random() < 0.85 else "2-line"

    if layout == "1-line":
        w, h = 420, 96
    else:
        w, h = 300, 180

    category = random.choices(
        ["private", "commercial", "ev", "rental"],
        weights=[0.65, 0.20, 0.10, 0.05],
        k=1
    )[0]

    if category == "private":
        bg_color = (random.randint(235, 250), random.randint(235, 250), random.randint(235, 250))
        text_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
    elif category == "commercial":
        bg_color = (random.randint(20, 50), random.randint(190, 225), random.randint(230, 255))
        text_color = (15, 15, 15)
    elif category == "ev":
        bg_color = (random.randint(30, 60), random.randint(130, 170), random.randint(30, 60))
        text_color = (250, 250, 250)
    else:
        bg_color = (random.randint(15, 30), random.randint(15, 30), random.randint(15, 30))
        text_color = (30, 210, 245)

    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Plate Border
    border_color = (25, 25, 25) if category != "rental" else (80, 80, 80)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), border_color, 2)

    # Blue HSRP strip
    if layout == "1-line":
        strip_w = 32
        cv2.rectangle(img, (2, 2), (strip_w, h - 3), (160, 70, 10), -1)
        cv2.rectangle(img, (9, 15), (23, 27), (180, 200, 210), -1)
        cv2.putText(img, "IND", (6, int(h * 0.70)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        text_x = strip_w + 14
    else:
        strip_w = 26
        cv2.rectangle(img, (2, 2), (strip_w, h - 3), (160, 70, 10), -1)
        cv2.putText(img, "IND", (4, int(h * 0.55)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
        text_x = strip_w + 12

    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
    font_scale = 1.15 if layout == "1-line" else 0.95
    thickness = 2

    if layout == "1-line":
        st = plate_text[:2]
        dist = plate_text[2:4]
        rest = plate_text[4:]
        if len(rest) > 4:
            series = rest[:-4]
            num = rest[-4:]
            formatted_text = f"{st} {dist} {series} {num}"
        else:
            formatted_text = f"{st} {dist} {rest}"

        text_y = int(h * 0.68)
        cv2.putText(img, formatted_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    else:
        st = plate_text[:2]
        dist = plate_text[2:4]
        line1 = f"{st} {dist}"
        line2 = plate_text[4:]
        cv2.putText(img, line1, (text_x, int(h * 0.42)), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(img, line2, (text_x, int(h * 0.85)), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Simulated Plate Bolts
    for bx in [strip_w + 20, w - 25]:
        by = int(h * 0.5)
        cv2.circle(img, (bx, by), 4, (120, 120, 120), -1)
        cv2.circle(img, (bx, by), 2, (40, 40, 40), -1)

    # Perspective Distortion / Camera Tilt
    if random.random() < 0.65:
        dx = random.randint(-25, 25)
        dy = random.randint(-15, 15)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [max(0, dx), max(0, dy)],
            [w - max(0, -dx), max(0, -dy)],
            [max(0, -dx), h - max(0, dy)],
            [w - max(0, dx), h - max(0, -dy)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Realistic Blur & Noise
    if random.random() < 0.40:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(5, 18), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    filename = f"svtr_syn_{idx:06d}_{plate_text}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img)

    return f"{filename}\t{plate_text}\n"

def generate_dataset(count: int = 50000, output_dir: str = "datasets/plate_ocr/synthetic_50k"):
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, "labels.txt")
    print(f"Generating {count} synthetic plates in {output_dir} using multi-threading...")

    lines = []
    # Batch processing with ThreadPool for fast IO
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate_single_plate, i, output_dir) for i in range(count)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            lines.append(future.result())
            if (i + 1) % 5000 == 0 or (i + 1) == count:
                print(f"  [{i + 1:05d}/{count}] Generated synthetic plates...")

    with open(labels_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Create dedicated train_svtr_50k.txt
    real_train = "datasets/plate_ocr/train.txt"
    svtr_train = "datasets/plate_ocr/train_svtr_50k.txt"
    merged_lines = []

    if os.path.exists(real_train):
        with open(real_train, "r", encoding="utf-8") as f:
            for line in f:
                merged_lines.append(line.strip() + "\n")
        print(f"Loaded {len(merged_lines)} real training crops from {real_train}")

    for l in lines:
        fn, txt = l.strip().split("\t")
        merged_lines.append(f"../synthetic_50k/{fn}\t{txt}\n")

    random.shuffle(merged_lines)
    with open(svtr_train, "w", encoding="utf-8") as f:
        f.writelines(merged_lines)

    print(f"[SUCCESS] Created {svtr_train} with {len(merged_lines)} total training samples!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--output", type=str, default="datasets/plate_ocr/synthetic_50k")
    args = parser.parse_args()
    generate_dataset(args.count, args.output)
