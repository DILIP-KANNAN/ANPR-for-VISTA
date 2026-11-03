"""
Training script for License Plate Detection using YOLOv8/YOLOv11 on RTX 4060 GPU.
Usage:
    python train_plate_detector.py --epochs 25 --batch 16 --imgsz 640
"""

import os
import argparse
import shutil
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train License Plate Detector on GPU")
    parser.add_argument("--data", type=str, default="datasets/plate_detection/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base pretrained model (yolov8n.pt, yolov8s.pt, yolov11n.pt)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (fits easily in 8GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="CUDA device or cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("      LICENSE PLATE DETECTOR - MODEL TRAINING")
    print("=" * 60)
    print(f"Base Model:    {args.model}")
    print(f"Dataset Config: {args.data}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch}")
    print(f"Image Size:    {args.imgsz}")
    print(f"Compute Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    # Load base YOLO model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        project="runs/detect",
        name="plate_detector",
        exist_ok=True,
        save=True,
        verbose=True
    )

    # Copy best weights to weights/best_plate.pt
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    best_pt = os.path.join("runs", "detect", "plate_detector", "weights", "best.pt")
    target_pt = os.path.join(weights_dir, "best_plate.pt")

    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        print(f"\n[SUCCESS] Best trained weights saved to: {os.path.abspath(target_pt)}")
    else:
        print("\nTraining completed. Weights saved in runs/detect/plate_detector/weights/")

if __name__ == "__main__":
    main()
