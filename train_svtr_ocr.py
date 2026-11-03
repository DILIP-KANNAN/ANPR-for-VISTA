"""
Training Script for SVTR-Tiny with STN Rectifier.
Trained on 51,467 Indian plate samples (50,000 synthetic + 1,467 real crops).
Uses Mixed Precision (AMP) on NVIDIA RTX 4060 GPU with CTC Loss.
Saves weights to weights/svtr_plate.pt (completely separate from CRNN).
"""

import os
import sys
import time
import argparse
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from anpr_pipeline.svtr_model import SVTRTinyWithSTN, decode_svtr_greedy

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 is CTC blank

class SVTRPlateDataset(Dataset):
    def __init__(self, txt_path: str, crops_dir: str, img_h: int = 32, img_w: int = 128, is_train: bool = True):
        self.crops_dir = crops_dir
        self.img_h = img_h
        self.img_w = img_w
        self.is_train = is_train

        self.samples = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    crop_file, text = parts[0], parts[1].upper()
                    text = "".join([c for c in text if c in CHAR2IDX])
                    if len(text) >= 4:
                        crop_p = os.path.join(crops_dir, crop_file)
                        if os.path.exists(crop_p):
                            self.samples.append((crop_p, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_path, text = self.samples[idx]
        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        if self.is_train:
            # Dynamic Lighting and Contrast Augmentations
            if random.random() < 0.5:
                alpha = random.uniform(0.75, 1.25)
                beta = random.randint(-20, 20)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

            # Random Occlusion Cutout Patch
            h_img, w_img = img.shape[:2]
            if random.random() < 0.25 and h_img >= 30 and w_img >= 60:
                rx = random.randint(5, w_img - 20)
                ry = random.randint(5, h_img - 15)
                rw, rh = random.randint(4, 12), random.randint(4, 12)
                img[ry:ry+rh, rx:rx+rw] = random.randint(40, 180)

        resized = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)
        tensor = ((torch.from_numpy(resized).float() / 127.5) - 1.0).unsqueeze(0)  # (1, 32, 128)
        label = [CHAR2IDX[c] for c in text]
        return tensor, torch.tensor(label, dtype=torch.long), text

def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels, 0)
    return images, targets, target_lengths, texts

def train_svtr():
    parser = argparse.ArgumentParser(description="Train SVTR-Tiny Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training SVTR-Tiny on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_set = SVTRPlateDataset("datasets/plate_ocr/train_svtr_50k.txt", "datasets/plate_ocr/crops", is_train=True)
    val_set = SVTRPlateDataset("datasets/plate_ocr/val.txt", "datasets/plate_ocr/crops", is_train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"Dataset: {len(train_set):,} training samples, {len(val_set)} validation crops", flush=True)

    model = SVTRTinyWithSTN(num_classes=len(CHARS) + 1, use_stn=True).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler()

    best_val_char_acc = 0.0
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/svtr_plate.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for imgs, targets, target_lens, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                preds = model(imgs)  # (T=32, B, C=37)
                t_len, b_sz, _ = preds.size()
                input_lens = torch.full(size=(b_sz,), fill_value=t_len, dtype=torch.long, device=device)
                log_probs = preds.log_softmax(2)
                loss = criterion(log_probs, targets.to(device), input_lens, target_lens.to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        em_count = 0
        total_chars = 0
        correct_chars = 0
        total_val = 0

        with torch.no_grad():
            for imgs, _, _, texts in val_loader:
                imgs = imgs.to(device)
                with autocast():
                    logits = model(imgs)
                decoded = decode_svtr_greedy(logits, CHARS)

                for pred_str, gt_str in zip(decoded, texts):
                    total_val += 1
                    if pred_str == gt_str:
                        em_count += 1
                    total_chars += max(len(gt_str), len(pred_str))
                    correct_chars += sum(1 for c1, c2 in zip(pred_str, gt_str) if c1 == c2)

        val_em = em_count / max(1, total_val) * 100.0
        val_char_acc = correct_chars / max(1, total_chars) * 100.0
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {avg_loss:.4f} | Val EM: {val_em:.2f}% | Val CharAcc: {val_char_acc:.2f}% | Time: {elapsed:.1f}s", flush=True)

        if val_char_acc > best_val_char_acc or epoch == args.epochs:
            best_val_char_acc = val_char_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "vocab": CHARS,
                "val_em": val_em,
                "val_char_acc": val_char_acc
            }, save_path)

    print(f"\n[SUCCESS] SVTR-Tiny Training finished! Best Val CharAcc: {best_val_char_acc:.2f}%. Saved to {save_path}")

if __name__ == "__main__":
    train_svtr()
