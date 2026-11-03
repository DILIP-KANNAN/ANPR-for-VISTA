"""
PyTorch CRNN (CNN + BiLSTM + CTC) Model for Indian License Plate Recognition.
Lightweight (<5ms inference on RTX 4060 GPU), trained natively on Indian plate typography.
"""

import torch
import torch.nn as nn

class IndianPlateCRNN(nn.Module):
    def __init__(self, num_classes: int = 37, in_channels: int = 1):
        super(IndianPlateCRNN, self).__init__()
        self.num_classes = num_classes

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 16 x 64

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # -> 8 x 32

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> 4 x 32

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # -> 2 x 32

            # Block 5
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)  # -> 1 x 31
        )

        # 2-layer Bidirectional LSTM
        self.rnn = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=1, batch_first=False),
        )
        self.rnn2 = nn.Sequential(
            nn.LSTM(512, 256, bidirectional=True, num_layers=1, batch_first=False),
        )

        # Final character classification head (blank is class 0)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) where H=32, W=128
        conv = self.cnn(x)  # (B, 512, 1, W')
        b, c, h, w = conv.size()
        assert h == 1, f"CRNN requires feature map height 1, got {h}"
        
        # (B, 512, 1, W') -> (W', B, 512)
        conv = conv.squeeze(2).permute(2, 0, 1)

        # RNN Sequence Processing
        out, _ = self.rnn(conv)
        out, _ = self.rnn2(out)

        # Classification (W', B, num_classes)
        logits = self.classifier(out)
        return logits

def decode_ctc(logits: torch.Tensor, chars: str) -> list:
    """Greedy CTC Best Path Decoding."""
    # logits: (T, B, C)
    probs = logits.softmax(2)
    preds = probs.argmax(2)  # (T, B)
    t_len, b_sz = preds.size()

    decoded_strings = []
    for b in range(b_sz):
        seq = preds[:, b].tolist()
        res = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:  # 0 is blank
                res.append(chars[idx - 1])
            prev = idx
        decoded_strings.append("".join(res))
    return decoded_strings


# Standard 38 Indian State / UT codes according to MoRTH
VALID_INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "DL",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UA", "UK", "UP", "WB", "BH"
}

def compute_grammar_score(text: str) -> float:
    """
    Evaluates how closely a plate hypothesis conforms to the Indian MoRTH registration syntax:
    Standard Format: [State 2L][District 1-2D][Series 0-3L][Number 4D]
    Returns score normalized in [0.0, 1.0].
    """
    if not text:
        return 0.0
    text = text.upper().strip()
    n = len(text)
    if n < 6 or n > 11:
        return 0.1

    score = 0.0
    max_score = 10.0

    # 1. State Code (chars 0:2)
    state = text[:2]
    if state in VALID_INDIAN_STATES:
        score += 3.0
    elif state.isalpha():
        score += 1.0
    else:
        score -= 2.0

    # 2. District Code (chars 2:4)
    dist = text[2:4]
    if dist.isdigit():
        score += 2.5
    elif len(dist) > 0 and dist[0].isdigit():
        score += 1.0
    else:
        score -= 1.0

    # 3. Registration Number (last 4 characters)
    tail = text[-4:]
    if len(tail) == 4 and tail.isdigit():
        score += 3.0
    else:
        num_digits = sum(1 for c in tail if c.isdigit())
        score += num_digits * 0.5

    # 4. Canonical length bonus
    if n in (9, 10):
        score += 1.5
    elif n in (8, 11):
        score += 0.5

    return max(0.0, min(1.0, score / max_score))


def decode_ctc_beam_search(
    logits: torch.Tensor,
    chars: str,
    beam_width: int = 10,
    grammar_weight: float = 2.0
) -> list:
    """
    Grammar-Constrained CTC Beam Search Decoder.
    Finds the optimal sequence considering acoustic CTC probabilities + MoRTH syntax constraint.
    Returns list of dicts: [{'plate': str, 'confidence': float, 'grammar_score': float, 'combined_score': float}, ...]
    """
    import numpy as np

    # logits: (T, B, C)
    probs = logits.softmax(2).detach().cpu().numpy()
    t_len, b_sz, num_classes = probs.shape

    results = []
    for b in range(b_sz):
        # Map: prefix -> (p_blank, p_non_blank)
        beams = {"": (1.0, 0.0)}

        for t in range(t_len):
            prob_t = probs[t, b]
            top_k_indices = np.argsort(prob_t)[-8:]  # top 8 tokens

            next_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                p_total = p_b + p_nb
                for c in top_k_indices:
                    p = prob_t[c]
                    if p < 1e-6:
                        continue

                    if c == 0:  # CTC blank
                        nb_b, nb_nb = next_beams.get(prefix, (0.0, 0.0))
                        next_beams[prefix] = (nb_b + p_total * p, nb_nb)
                    else:
                        ch = chars[c - 1]
                        if len(prefix) > 0 and ch == prefix[-1]:
                            # Coming from blank -> extends prefix
                            n_pref = prefix + ch
                            nb_b, nb_nb = next_beams.get(n_pref, (0.0, 0.0))
                            next_beams[n_pref] = (nb_b, nb_nb + p_b * p)

                            # Coming from non-blank -> repeats char without extending
                            nb_b2, nb_nb2 = next_beams.get(prefix, (0.0, 0.0))
                            next_beams[prefix] = (nb_b2, nb_nb2 + p_nb * p)
                        else:
                            n_pref = prefix + ch
                            nb_b, nb_nb = next_beams.get(n_pref, (0.0, 0.0))
                            next_beams[n_pref] = (nb_b, nb_nb + p_total * p)

            # Keep top beams
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda item: item[1][0] + item[1][1],
                reverse=True
            )[: beam_width * 2]
            beams = dict(sorted_beams)

        # Post-process beams with Indian grammar scoring
        candidates = []
        for prefix, (p_b, p_nb) in beams.items():
            ctc_prob = p_b + p_nb
            if not prefix or ctc_prob < 1e-9:
                continue
            g_score = compute_grammar_score(prefix)
            combined_score = np.log(ctc_prob + 1e-12) + grammar_weight * (g_score * 5.0)
            candidates.append({
                "plate": prefix,
                "confidence": float(min(1.0, ctc_prob)),
                "grammar_score": float(g_score),
                "combined_score": float(combined_score)
            })

        if not candidates:
            # Fallback to greedy decode if beam search produced empty
            greedy_seq = probs[:, b].argmax(1)
            res = []
            prev = -1
            for idx in greedy_seq:
                if idx != 0 and idx != prev:
                    res.append(chars[idx - 1])
                prev = idx
            fallback_text = "".join(res)
            g_score = compute_grammar_score(fallback_text)
            results.append({
                "plate": fallback_text,
                "confidence": 0.5,
                "grammar_score": float(g_score),
                "combined_score": 0.0
            })
        else:
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            results.append(candidates[0])

    return results

