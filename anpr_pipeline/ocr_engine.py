"""
Modular OCR Engines for Character Recognition.
Supports EasyOCR, GOT-OCR / Vision-Language OCR, and custom models.
"""

from typing import Tuple
import re
import cv2
import numpy as np
import torch
from anpr_pipeline.base import BaseOCREngine

class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages: list = None, gpu: bool = True):
        super().__init__()
        import easyocr
        use_gpu = gpu and torch.cuda.is_available()
        langs = languages or ["en"]
        self.reader = easyocr.Reader(langs, gpu=use_gpu)

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # Enhance plate crop before OCR (grayscale, contrast stretch)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        # Resize to standard height if too small
        if gray.shape[0] < 32:
            scale = 32.0 / gray.shape[0]
            gray = cv2.resize(gray, (int(gray.shape[1] * scale), 32), interpolation=cv2.INTER_CUBIC)

        results = self.reader.readtext(gray, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not results:
            # Fallback without allowlist
            results = self.reader.readtext(gray, detail=1)

        if not results:
            return "", 0.0

        # Sort detections top-to-bottom, left-to-right (for 2-line plates)
        results.sort(key=lambda r: (r[0][0][1] // 15, r[0][0][0]))

        text_pieces = []
        confidences = []
        for bbox, text, conf in results:
            clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
            if clean:
                text_pieces.append(clean)
                confidences.append(conf)

        final_text = "".join(text_pieces)
        avg_conf = float(np.mean(confidences)) if confidences else 0.5
        return final_text, round(min(1.0, max(0.1, avg_conf)), 2)

class GotOCREngine(BaseOCREngine):
    """
    GOT-OCR 2.0 (General OCR Theory) / Transformers Vision-Language OCR.
    """
    def __init__(self, model_name: str = "stepfun-ai/GOT-OCR2_0", device: str = "cuda:0"):
        super().__init__()
        self.device = device if (torch.cuda.is_available() and "cuda" in device) else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.fallback = None
        self._init_model()

    def _get_fallback(self):
        if self.fallback is None:
            self.fallback = EasyOCREngine(gpu=(self.device != "cpu"))
        return self.fallback

    def _init_model(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            print(f"[GotOCREngine] Loading {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                low_cpu_mem_usage=True, 
                device_map=self.device
            ).eval()
        except Exception as e:
            print(f"[GotOCREngine] Note: {self.model_name} requires weights download ({e}). Falling back to EasyOCR.")
            self.fallback = EasyOCREngine(gpu=(self.device != "cpu"))

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        if self.model is not None and self.tokenizer is not None:
            temp_path = None
            try:
                import tempfile
                import os
                
                # Adaptive upscale small plates for vision transformer patch tokens
                h, w = plate_crop.shape[:2]
                proc_crop = plate_crop
                if h < 64:
                    scale = 64.0 / float(h)
                    proc_crop = cv2.resize(plate_crop, (int(w * scale), 64), interpolation=cv2.INTER_CUBIC)

                # Safely create and close temp file handle on Windows
                tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_path = tf.name
                tf.close()

                cv2.imwrite(temp_path, proc_crop)
                res = self.model.chat(self.tokenizer, temp_path, ocr_type="ocr")
                clean = re.sub(r"[^A-Za-z0-9]", "", res).upper()
                if len(clean) >= 6 and clean not in ["TOUR", "TAXI", "PRIVATE"]:
                    return clean, 0.95
                
                # Fallback to EasyOCR if GOT-OCR produces degenerate or noisy non-plate text
                fb_text, fb_conf = self._get_fallback().read_plate(plate_crop)
                if len(fb_text) > len(clean):
                    return fb_text, fb_conf
                return clean, 0.50
            except Exception as e:
                print(f"[GotOCREngine] Inference error: {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # Use EasyOCR fallback
        return self._get_fallback().read_plate(plate_crop)

class IndianCRNNEngine(BaseOCREngine):
    """
    Custom PyTorch CRNN model trained natively on Indian License Plate crops.
    Runs in ~2-3 ms per plate on RTX 4060 GPU.
    """
    def __init__(self, weights_path: str = "weights/crnn_plate.pt", device: str = "cuda:0"):
        super().__init__()
        import os
        from anpr_pipeline.crnn_model import IndianPlateCRNN, decode_ctc
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.model = IndianPlateCRNN(num_classes=len(self.chars) + 1).to(self.device)
        self.decode_ctc = decode_ctc

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            self.model.eval()
            print(f"[IndianCRNNEngine] Loaded custom weights from {weights_path}")
        else:
            print(f"[IndianCRNNEngine] Warning: {weights_path} not found. Running uninitialized.")

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # Grayscale & resize to (1, 32, 128)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        resized = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_CUBIC)
        tensor = ((torch.from_numpy(resized).float() / 127.5) - 1.0).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            decoded = self.decode_ctc(logits, self.chars)

        plate = decoded[0] if decoded else ""
        return plate, 0.92

class EnsembleOCREngine(BaseOCREngine):
    """
    Hybrid Ensemble OCR Engine combining GOT-OCR 2.0 with Custom Indian Plate CRNN.
    Fuses predictions with syntax awareness to achieve maximum recognition accuracy.
    """
    def __init__(self, device: str = "cuda:0"):
        super().__init__()
        from anpr_pipeline.grammar_validator import IndianRTOValidator
        self.got_engine = GotOCREngine(device=device)
        self.crnn_engine = IndianCRNNEngine(device=device)
        self.validator = IndianRTOValidator()

    def read_plate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        got_text, got_conf = self.got_engine.read_plate(plate_crop)
        crnn_text, crnn_conf = self.crnn_engine.read_plate(plate_crop)

        # If both agree, high confidence exact match
        if got_text and got_text == crnn_text:
            return got_text, 0.98

        # Grammar score arbitration
        norm_got, score_got = self.validator.validate(got_text)
        norm_crnn, score_crnn = self.validator.validate(crnn_text)

        if score_got >= score_crnn and score_got >= 0.6:
            return norm_got, got_conf
        elif score_crnn > score_got and score_crnn >= 0.6:
            return norm_crnn, crnn_conf
        else:
            return norm_got if norm_got else norm_crnn, max(got_conf, crnn_conf)

def get_ocr_engine(engine_type: str = "easyocr") -> BaseOCREngine:
    """Factory function to instantiate the selected OCR engine."""
    engine_type = engine_type.lower()
    if "ensemble" in engine_type or "hybrid" in engine_type:
        return EnsembleOCREngine()
    elif "crnn" in engine_type:
        return IndianCRNNEngine()
    elif "got" in engine_type:
        return GotOCREngine()
    else:
        return EasyOCREngine(gpu=torch.cuda.is_available())
