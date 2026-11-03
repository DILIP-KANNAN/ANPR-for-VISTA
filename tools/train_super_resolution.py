"""
Training script for PlateESPCN Super-Resolution on Indian License Plate Crops.
Learns character-edge reconstruction specifically tailored for license plate typography.
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Windows OpenMP protection
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anpr_pipeline.super_resolution import PlateESPCN

class PlateSRDataset(Dataset):
    def __init__(self, crops_dir: str = "datasets/plate_ocr/crops", split_file: str = "datasets/plate_ocr/train.txt", hr_height: int = 64, hr_width: int = 192):
        self.hr_h = hr_height
        self.hr_w = hr_width
        self.image_paths = []

        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        p = os.path.join(crops_dir, parts[0])
                        if os.path.exists(p):
                            self.image_paths.append(p)
        else:
            self.image_paths = glob.glob(os.path.join(crops_dir, "*.jpg"))

        print(f"[PlateSRDataset] Loaded {len(self.image_paths)} plate training images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.hr_h, self.hr_w, 3), dtype=np.uint8)

        # Convert to Y channel
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]

        # High-res ground truth
        hr_y = cv2.resize(y, (self.hr_w, self.hr_h), interpolation=cv2.INTER_CUBIC)

        # Low-res input: downscaled by 0.5x with subtle blur
        lr_h, lr_w = self.hr_h // 2, self.hr_w // 2
        lr_y = cv2.resize(hr_y, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
        if np.random.rand() > 0.5:
            lr_y = cv2.GaussianBlur(lr_y, (3, 3), 0.5)

        # Normalize to [0.0, 1.0]
        lr_tensor = torch.from_numpy(lr_y.astype(np.float32) / 255.0).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_y.astype(np.float32) / 255.0).unsqueeze(0)

        return lr_tensor, hr_tensor


class SobelEdgeLoss(nn.Module):
    """Encourages crisp, sharp alphanumeric strokes."""
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(sobel_y, requires_grad=False)
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        px, py = self.conv_x(pred), self.conv_y(pred)
        tx, ty = self.conv_x(target), self.conv_y(target)
        return self.l1(px, tx) + self.l1(py, ty)


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[PlateESPCN] Training on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset = PlateSRDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = PlateESPCN(upscale_factor=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    l1_loss_fn = nn.L1Loss()
    edge_loss_fn = SobelEdgeLoss(device)

    epochs = 20
    print(f"[PlateESPCN] Starting {epochs} epochs of training...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)

            l1 = l1_loss_fn(sr, hr)
            edge = edge_loss_fn(sr, hr)
            loss = l1 + 0.3 * edge

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch [{ep:2d}/{epochs:2d}] | Loss: {avg_loss:.5f} (L1 + Edge)")

    total_time = time.time() - t0
    print(f"[PlateESPCN] Training completed in {total_time:.1f}s.")

    os.makedirs("weights", exist_ok=True)
    out_path = "weights/espcn_plate.pt"
    torch.save(model.state_dict(), out_path)
    print(f"[PlateESPCN] Saved trained model weights to: {out_path}")

if __name__ == "__main__":
    train()
