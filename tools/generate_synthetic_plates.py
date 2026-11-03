"""
Synthetic Indian License Plate Generator.
Generates photorealistic 1-Line and 2-Line Indian plates with diverse plate classes,
fonts, HSRP security features, and camera distortions.
"""

import os
import random
import string
import argparse
import cv2
import numpy as np

INDIAN_STATES = [
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", 
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", 
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", 
    "WB", "BH", "OR"
]

# Prohibited letters in MoRTH series: I and O
VALID_SERIES_LETTERS = [c for c in string.ascii_uppercase if c not in ("I", "O")]

def random_plate_number():
    state = random.choice(INDIAN_STATES)
    rto = f"{random.randint(1, 99):02d}"
    plate_type = random.random()
    if plate_type < 0.70:
        # Standard 10-char (2 state + 2 rto + 2 series + 4 num)
        series = "".join(random.choices(VALID_SERIES_LETTERS, k=2))
        digits = f"{random.randint(1, 9999):04d}"
    elif plate_type < 0.90:
        # 9-char (2 state + 2 rto + 1 series + 4 num)
        series = "".join(random.choices(VALID_SERIES_LETTERS, k=1))
        digits = f"{random.randint(1, 9999):04d}"
    else:
        # Vintage 8-char (2 state + 2 rto + 4 num)
        series = ""
        digits = f"{random.randint(1, 9999):04d}"

    return f"{state}{rto}{series}{digits}", state

def create_plate_image(plate_text: str, layout: str = "1-line"):
    if layout == "1-line":
        w, h = 420, 96
    else:  # 2-line stacked
        w, h = 300, 180

    # Plate Category & Color Themes (BGR)
    category = random.choices(
        ["private", "commercial", "ev", "rental"],
        weights=[0.65, 0.20, 0.10, 0.05],
        k=1
    )[0]

    if category == "private":
        bg_color = (random.randint(235, 250), random.randint(235, 250), random.randint(235, 250))
        text_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
    elif category == "commercial":
        bg_color = (random.randint(20, 50), random.randint(190, 225), random.randint(230, 255))  # Yellow
        text_color = (15, 15, 15)
    elif category == "ev":
        bg_color = (random.randint(30, 60), random.randint(130, 170), random.randint(30, 60))  # Green
        text_color = (250, 250, 250)
    else:  # rental
        bg_color = (random.randint(15, 30), random.randint(15, 30), random.randint(15, 30))  # Black
        text_color = (30, 210, 245)  # Yellow text

    img = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Plate Outer Border
    border_color = (25, 25, 25) if category != "rental" else (80, 80, 80)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), border_color, 2)

    # Blue HSRP strip on the left with chromium hologram & 'IND'
    if layout == "1-line":
        strip_w = 32
        cv2.rectangle(img, (2, 2), (strip_w, h - 3), (160, 70, 10), -1)  # Blue in BGR
        # Chromium hologram simulation
        cv2.rectangle(img, (9, 15), (23, 27), (180, 200, 210), -1)
        cv2.putText(img, "IND", (6, int(h * 0.70)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        text_x = strip_w + 14
    else:
        strip_w = 26
        cv2.rectangle(img, (2, 2), (strip_w, h - 3), (160, 70, 10), -1)
        cv2.putText(img, "IND", (4, int(h * 0.55)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
        text_x = strip_w + 12

    # Render alphanumeric plate text
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
    font_scale = 1.15 if layout == "1-line" else 0.95
    thickness = 2

    if layout == "1-line":
        # Formatted spacing: e.g. "MH 01 BT 7906" or "PB 35 F 9333"
        st = plate_text[:2]
        dist = plate_text[2:4]
        rest = plate_text[4:]
        if len(rest) > 4:
            series = rest[:-4]
            num = rest[-4:]
            formatted = f"{st} {dist} {series} {num}"
        else:
            formatted = f"{st} {dist} {rest}"

        cv2.putText(img, formatted, (text_x, int(h * 0.70)), font, font_scale, text_color, thickness, cv2.LINE_AA)
    else:
        # Stacked: line 1 = State + District (+ Series), line 2 = Number
        if len(plate_text) >= 8:
            line1 = plate_text[:-4]
            line2 = plate_text[-4:]
        else:
            line1 = plate_text[:4]
            line2 = plate_text[4:]

        cv2.putText(img, line1, (text_x, int(h * 0.44)), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(img, line2, (text_x + 15, int(h * 0.88)), font, font_scale * 1.1, text_color, thickness + 1, cv2.LINE_AA)

    # Real-World Camera & Physical Distortions
    # 1. Perspective slant/tilt
    dx = random.randint(-12, 12)
    dy = random.randint(-8, 8)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[dx, dy], [w - dx, -dy], [-dx, h + dy], [w + dx, h - dy]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h), borderValue=bg_color)

    # 2. Random mounting bolt/screw circles (occlusions)
    if random.random() < 0.6:
        bolt_x1 = random.randint(strip_w + 10, strip_w + 60)
        bolt_x2 = random.randint(w - 70, w - 20)
        bolt_y = random.randint(12, h - 12)
        cv2.circle(img, (bolt_x1, bolt_y), random.randint(4, 6), (100, 100, 100), -1)
        cv2.circle(img, (bolt_x2, bolt_y), random.randint(4, 6), (100, 100, 100), -1)

    # 3. Illumination gradient / shadow
    if random.random() < 0.5:
        gradient = np.linspace(0.8, 1.2, w).reshape(1, w, 1)
        img = np.clip(img * gradient, 0, 255).astype(np.uint8)

    # 4. Blur / defocus
    if random.random() < 0.4:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 5. Sensor noise
    if random.random() < 0.4:
        noise = np.random.normal(0, random.randint(4, 12), img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return img

def generate_dataset(count: int = 6000, output_dir: str = "datasets/plate_ocr/synthetic_v2"):
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, "labels.txt")

    print(f"Generating {count} photorealistic synthetic Indian plates into {output_dir}...")
    lines = []
    for i in range(count):
        plate_text, state = random_plate_number()
        layout = "1-line" if random.random() < 0.85 else "2-line"
        img = create_plate_image(plate_text, layout=layout)
        filename = f"syn_{i:06d}_{plate_text}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        lines.append(f"{filename}\t{plate_text}\n")

        if (i + 1) % 1000 == 0 or (i + 1) == count:
            print(f"  [{i + 1}/{count}] Generated...")

    with open(labels_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[DONE] Generated {count} synthetic plates. Labels saved to {labels_file}")

    # Merge synthetic dataset with existing real train.txt
    real_train = "datasets/plate_ocr/train.txt"
    aug_train = "datasets/plate_ocr/train_augmented.txt"
    merged_lines = []

    if os.path.exists(real_train):
        with open(real_train, "r", encoding="utf-8") as f:
            for line in f:
                # Real crops are in datasets/plate_ocr/crops/
                merged_lines.append(line.strip() + "\n")
        print(f"Loaded {len(merged_lines)} real training crops from {real_train}")

    # Append synthetic lines with path prefix
    for l in lines:
        fn, txt = l.strip().split("\t")
        merged_lines.append(f"../synthetic_v2/{fn}\t{txt}\n")

    random.shuffle(merged_lines)
    with open(aug_train, "w", encoding="utf-8") as f:
        f.writelines(merged_lines)

    print(f"Successfully created {aug_train} with {len(merged_lines)} total training samples!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6000, help="Number of plates to generate")
    parser.add_argument("--output", type=str, default="datasets/plate_ocr/synthetic_v2", help="Output directory")
    args = parser.parse_args()
    generate_dataset(args.count, args.output)

if __name__ == "__main__":
    main()
