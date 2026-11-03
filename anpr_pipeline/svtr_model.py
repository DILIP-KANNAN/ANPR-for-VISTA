"""
SVTR-Tiny Architecture with Spatial Transformer Network (STN) Rectifier.
Combines:
  1. STN Rectifier: Differentiable affine rectification for tilted / perspective-skewed plates.
  2. SVTR-Tiny: Single Visual Model for Text Recognition (3.8M params) with local stroke mixing and global context.
  3. MoRTH Grammar-Constrained Beam Search Decoder.
"""

import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from anpr_pipeline.crnn_model import VALID_INDIAN_STATES, compute_grammar_score

# -------------------------------------------------------------------------
# 1. Spatial Transformer Network (STN) Rectifier
# -------------------------------------------------------------------------

class STNRectifier(nn.Module):
    """
    Differentiable Spatial Transformer Network (STN).
    Predicts an affine transformation matrix theta (2x3) to dynamically unwrap
    and horizontally rectify tilted, skewed, or angled license plate crops.
    """
    def __init__(self, in_channels: int = 1):
        super(STNRectifier, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 16 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 8 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 4 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((2, 4))                                      # 128 x 2 x 4 = 1024
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 2 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)
        )

        # Initialize the weights/bias to identity affine transformation:
        # [1, 0, 0]
        # [0, 1, 0]
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rectified = F.grid_sample(x, grid, align_corners=False)
        return rectified, theta

# -------------------------------------------------------------------------
# 2. SVTR Mixing Blocks (Local & Global Visual Self-Attention)
# -------------------------------------------------------------------------

class ConvBNLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SVTRBlock(nn.Module):
    """
    SVTR Mixing Block with either:
      - Local Mixing: 2D depthwise convolution modeling character stroke geometries
      - Global Mixing: Multi-Head Self Attention across the character sequence
    """
    def __init__(self, dim: int, num_heads: int = 4, mixing_type: str = "local", mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.mixing_type = mixing_type
        self.norm1 = nn.LayerNorm(dim)

        if mixing_type == "local":
            # Local 2D depthwise convolution for stroke interaction
            self.mixer = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=(5, 5), padding=(2, 2), groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim)
            )
        else:
            # Global Multi-Head Self-Attention
            self.mixer = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: (B, N, C) where N = h * w
        b, n, c = x.shape
        residual = x
        x_norm = self.norm1(x)

        if self.mixing_type == "local":
            # Reshape to (B, C, H, W) for 2D depthwise convolution
            feat = x_norm.transpose(1, 2).view(b, c, h, w)
            feat = self.mixer(feat)
            feat = feat.view(b, c, n).transpose(1, 2)
            x = residual + feat
        else:
            # Multihead Attention
            attn_out, _ = self.mixer(x_norm, x_norm, x_norm)
            x = residual + attn_out

        x = x + self.mlp(self.norm2(x))
        return x

# -------------------------------------------------------------------------
# 3. Complete SVTR-Tiny Model with STN Rectifier
# -------------------------------------------------------------------------

class SVTRTinyWithSTN(nn.Module):
    """
    SVTR-Tiny Architecture for License Plate Recognition:
      - Parameters: ~3.8M
      - Resolution: 32x128 -> Rectified -> 32 character sequence tokens
      - Output: (T=32, B, num_classes)
    """
    def __init__(self, num_classes: int = 37, in_channels: int = 1, use_stn: bool = True):
        super().__init__()
        self.use_stn = use_stn
        self.num_classes = num_classes

        # Optional STN Rectifier
        if use_stn:
            self.stn = STNRectifier(in_channels=in_channels)
        else:
            self.stn = None

        # Patch Embedding: 32x128 -> 8x32 with 128 channels
        self.patch_embed = nn.Sequential(
            ConvBNLayer(in_channels, 64, kernel_size=3, stride=2, padding=1),   # 16 x 64
            ConvBNLayer(64, 128, kernel_size=3, stride=2, padding=1)           # 8 x 32
        )

        # Stage 1: 2 Local Mixing Blocks (dim=128)
        self.stage1 = nn.ModuleList([
            SVTRBlock(dim=128, mixing_type="local") for _ in range(2)
        ])

        # Subsampling 1: 8x32 -> 4x32 (dim=256)
        self.sub1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # Stage 2: 3 Mixed Blocks (dim=256)
        self.stage2 = nn.ModuleList([
            SVTRBlock(dim=256, mixing_type="local"),
            SVTRBlock(dim=256, mixing_type="global"),
            SVTRBlock(dim=256, mixing_type="local")
        ])

        # Subsampling 2: 4x32 -> 2x32 (dim=384)
        self.sub2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(384),
            nn.GELU()
        )

        # Stage 3: 2 Global Mixing Blocks (dim=384)
        self.stage3 = nn.ModuleList([
            SVTRBlock(dim=384, mixing_type="global") for _ in range(2)
        ])

        # Height pool: 2x32 -> 1x32 (dim=384)
        self.height_pool = nn.AdaptiveAvgPool2d((1, 32))
        self.dropout = nn.Dropout(0.15)

        # Classification Head (T=32, B, num_classes)
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, 128)
        if self.stn is not None:
            x, _ = self.stn(x)

        # Patch Embedding -> (B, 128, 8, 32)
        feat = self.patch_embed(x)
        b, c, h, w = feat.shape

        # Stage 1
        tokens = feat.flatten(2).transpose(1, 2)  # (B, 256, 128)
        for blk in self.stage1:
            tokens = blk(tokens, h, w)

        # Subsample 1 -> (B, 256, 4, 32)
        feat = tokens.transpose(1, 2).view(b, c, h, w)
        feat = self.sub1(feat)
        b, c, h, w = feat.shape

        # Stage 2
        tokens = feat.flatten(2).transpose(1, 2)
        for blk in self.stage2:
            tokens = blk(tokens, h, w)

        # Subsample 2 -> (B, 384, 2, 32)
        feat = tokens.transpose(1, 2).view(b, c, h, w)
        feat = self.sub2(feat)
        b, c, h, w = feat.shape

        # Stage 3
        tokens = feat.flatten(2).transpose(1, 2)
        for blk in self.stage3:
            tokens = blk(tokens, h, w)

        # Pool height to 1 -> (B, 384, 1, 32)
        feat = tokens.transpose(1, 2).view(b, c, h, w)
        feat = self.height_pool(feat).squeeze(2)  # (B, 384, 32)

        # Sequence representation (T=32, B, 384)
        seq = feat.permute(2, 0, 1)
        seq = self.dropout(seq)

        # Logits (T=32, B, num_classes)
        logits = self.classifier(seq)
        return logits

# -------------------------------------------------------------------------
# 4. Decoders (Greedy and Grammar-Constrained Beam Search)
# -------------------------------------------------------------------------

def decode_svtr_greedy(logits: torch.Tensor, chars: str) -> List[str]:
    """Greedy argmax CTC decoding for SVTR."""
    probs = logits.softmax(2)
    preds = probs.argmax(2)  # (T, B)
    t_len, b_sz = preds.size()

    decoded = []
    for b in range(b_sz):
        seq = preds[:, b].tolist()
        res = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:
                res.append(chars[idx - 1])
            prev = idx
        decoded.append("".join(res))
    return decoded


def decode_svtr_beam_search(
    logits: torch.Tensor,
    chars: str,
    beam_width: int = 10,
    grammar_weight: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Grammar-Constrained CTC Beam Search Decoder for SVTR-Tiny.
    Returns: [{'plate': str, 'confidence': float, 'grammar_score': float, 'combined_score': float}, ...]
    """
    probs = logits.softmax(2).detach().cpu().numpy()
    t_len, b_sz, num_classes = probs.shape

    results = []
    for b in range(b_sz):
        beams = {"": (1.0, 0.0)}

        for t in range(t_len):
            prob_t = probs[t, b]
            top_k = np.argsort(prob_t)[-8:]

            next_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_tot = p_b + p_nb
                for c in top_k:
                    p = prob_t[c]
                    if p < 1e-6:
                        continue

                    if c == 0:  # Blank
                        nb_b, nb_nb = next_beams.get(prefix, (0.0, 0.0))
                        next_beams[prefix] = (nb_b + p_tot * p, nb_nb)
                    else:
                        ch = chars[c - 1]
                        if len(prefix) > 0 and ch == prefix[-1]:
                            n_pref = prefix + ch
                            nb_b, nb_nb = next_beams.get(n_pref, (0.0, 0.0))
                            next_beams[n_pref] = (nb_b, nb_nb + p_b * p)

                            nb_b2, nb_nb2 = next_beams.get(prefix, (0.0, 0.0))
                            next_beams[prefix] = (nb_b2, nb_nb2 + p_nb * p)
                        else:
                            n_pref = prefix + ch
                            nb_b, nb_nb = next_beams.get(n_pref, (0.0, 0.0))
                            next_beams[n_pref] = (nb_b, nb_nb + p_tot * p)

            sorted_beams = sorted(
                next_beams.items(),
                key=lambda item: item[1][0] + item[1][1],
                reverse=True
            )[: beam_width * 2]
            beams = dict(sorted_beams)

        candidates = []
        for prefix, (p_b, p_nb) in beams.items():
            ctc_p = p_b + p_nb
            if not prefix or ctc_p < 1e-9:
                continue
            g_score = compute_grammar_score(prefix)
            combined = np.log(ctc_p + 1e-12) + grammar_weight * (g_score * 5.0)
            candidates.append({
                "plate": prefix,
                "confidence": float(min(1.0, ctc_p)),
                "grammar_score": float(g_score),
                "combined_score": float(combined)
            })

        if not candidates:
            greedy = probs[:, b].argmax(1)
            res = []
            prev = -1
            for idx in greedy:
                if idx != 0 and idx != prev:
                    res.append(chars[idx - 1])
                prev = idx
            txt = "".join(res)
            results.append({
                "plate": txt,
                "confidence": 0.5,
                "grammar_score": compute_grammar_score(txt),
                "combined_score": 0.0
            })
        else:
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            results.append(candidates[0])

    return results
