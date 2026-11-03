"""
Edge Deployment & Model Export Tool.
Exports YOLOv8 Plate Detector, CRNN Plate OCR, and OSNet Vehicle ReID
to TorchScript (.pt) and ONNX (.onnx) for deployment on NVIDIA Jetson (Orin Nano / NX)
and NVIDIA DeepStream SDK.
"""

import os
import sys
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anpr_pipeline.crnn_model import IndianPlateCRNN
from anpr_pipeline.vehicle_reid import OSNetBackbone

def export_crnn_model():
    print("\n[1/3] Exporting CRNN Plate OCR Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IndianPlateCRNN(num_classes=37).to(device).eval()

    weights_path = "weights/crnn_plate.pt"
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

    # Dummy tensor with dynamic batch size
    dummy_input = torch.randn(1, 1, 32, 128, device=device)

    # 1. TorchScript Export
    ts_path = "weights/crnn_plate.torchscript.pt"
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(ts_path)
        print(f"  -> TorchScript exported: {ts_path} ({os.path.getsize(ts_path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  -> TorchScript export error: {e}")

    # 2. ONNX Export with Dynamic Batching
    onnx_path = "weights/crnn_plate.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {1: "batch_size"}
            }
        )
        print(f"  -> ONNX model exported: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  -> ONNX export error: {e}")

def export_plate_detector():
    print("\n[2/3] Exporting YOLOv8 Plate Detector to ONNX...")
    weights_path = "weights/best_plate.pt"
    if not os.path.exists(weights_path):
        print(f"  -> Error: {weights_path} not found.")
        return

    try:
        yolo_model = YOLO(weights_path)
        yolo_model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
        onnx_file = "weights/best_plate.onnx"
        if os.path.exists(onnx_file):
            print(f"  -> YOLOv8 ONNX exported: {onnx_file} ({os.path.getsize(onnx_file) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  -> YOLO export error: {e}")

def export_reid_model():
    print("\n[3/3] Exporting OSNet Vehicle ReID Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_backbone = OSNetBackbone(num_features=512).to(device).eval()

    dummy_crop = torch.randn(1, 3, 128, 128, device=device)

    # 1. TorchScript Export
    ts_path = "weights/vehicle_reid.torchscript.pt"
    try:
        traced_reid = torch.jit.trace(reid_backbone, dummy_crop)
        traced_reid.save(ts_path)
        print(f"  -> TorchScript exported: {ts_path} ({os.path.getsize(ts_path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  -> TorchScript export error: {e}")

    # 2. ONNX Export
    onnx_path = "weights/vehicle_reid.onnx"
    try:
        torch.onnx.export(
            reid_backbone,
            dummy_crop,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["vehicle_crops"],
            output_names=["embeddings"],
            dynamic_axes={
                "vehicle_crops": {0: "batch_size"},
                "embeddings": {0: "batch_size"}
            }
        )
        print(f"  -> ONNX model exported: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  -> ONNX export error: {e}")

def main():
    print("=" * 75)
    print("        EDGE DEPLOYMENT & ONNX / TENSORRT EXPORT UTILITY")
    print("=" * 75)
    os.makedirs("weights", exist_ok=True)

    export_crnn_model()
    export_plate_detector()
    export_reid_model()

    print("\n" + "=" * 75)
    print("                  NVIDIA TENSORRT COMPILATION COMMANDS")
    print("=" * 75)
    print("To compile for NVIDIA Jetson Orin Nano/NX or RTX GPUs with TensorRT:")
    print("1. YOLOv8 Plate Detector (FP16):")
    print("   trtexec --onnx=weights/best_plate.onnx --fp16 --saveEngine=weights/best_plate.engine")
    print("\n2. Fast CRNN OCR (Dynamic Batch 1-16, FP16):")
    print("   trtexec --onnx=weights/crnn_plate.onnx --fp16 --minShapes=input:1x1x32x128 --optShapes=input:8x1x32x128 --maxShapes=input:16x1x32x128 --saveEngine=weights/crnn_plate.engine")
    print("\n3. OSNet Vehicle ReID (FP16):")
    print("   trtexec --onnx=weights/vehicle_reid.onnx --fp16 --saveEngine=weights/vehicle_reid.engine")
    print("=" * 75)

if __name__ == "__main__":
    main()
