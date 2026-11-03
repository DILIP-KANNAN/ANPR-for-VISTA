"""
Grammar Validator & Rule-based Corrector for Indian Registration Plates.
Follows standard MoRTH / RTO syntax: [State 2-char][District 2-digit][Series 1-3 char][Number 1-4 digit]
"""

import re
import numpy as np
from typing import Tuple, Optional
from anpr_pipeline.base import BaseGrammarValidator

# All valid Indian States & Union Territories registration codes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", 
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", 
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", 
    "WB", "BH", "OR"  # BH = Bharat series, OR = Orissa legacy
}

# Character confusion matrix for OCR post-processing
LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "D": "0", "A": "4", "Q": "0", "U": "0", "C": "0"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "3": "E", "7": "T"}

# Common OCR confusion for 2-letter state codes
STATE_CONFUSIONS = {
    "HH": "MH", "HL": "ML", "BB": "PB", "8R": "BR", "OH": "CH", 
    "OI": "DL", "0L": "DL", "1N": "TN", "TM": "TN", "UR": "UP", "UN": "UP",
    "RH": "MH", "MR": "MH", "MI": "MH", "M1": "MH", "B1": "MH", "BI": "MH",
    "AH": "MH", "0D": "OD", "AP": "AP"
}

from anpr_pipeline.character_ambiguity import CharacterAmbiguityResolver

class IndianRTOValidator(BaseGrammarValidator):
    def __init__(self):
        super().__init__()
        self.ambiguity_resolver = CharacterAmbiguityResolver()

    def validate(self, text: str, plate_crop: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Cleans, corrects, and scores an alphanumeric plate against Indian RTO grammar.
        Optionally uses plate_crop for secondary visual ambiguity verification.
        Returns: (normalized_plate, grammar_score [0.0 - 1.0])
        """
        if not text:
            return "", 0.0

        # Remove spaces, hyphens, special symbols, and convert to uppercase
        clean = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if len(clean) < 4:
            return clean, 0.2

        # 1. Embedded regex search ONLY if string contains extra trailing/leading noise beyond 10 chars
        if len(clean) > 10:
            m = re.search(r"([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4})", clean)
            if m and m.group(1)[:2] in INDIAN_STATE_CODES:
                clean = m.group(1)

        # 2. Strip leading HSRP prefix noise ('IND', '1ND', 'ND')
        if clean.startswith(("IND", "1ND")) and len(clean) >= 8:
            clean = clean[3:]
        elif clean.startswith("ND") and len(clean) >= 9 and clean[:2] not in INDIAN_STATE_CODES:
            clean = clean[2:]
        elif len(clean) >= 8 and clean[:2] not in INDIAN_STATE_CODES:
            # Check if 1st char is noise (e.g. CAP31DT9378 -> AP31DT9378)
            if clean[1:3] in INDIAN_STATE_CODES and (clean[3].isdigit() or clean[3] in LETTER_TO_DIGIT):
                clean = clean[1:]
            elif len(clean) >= 9 and clean[2:4] in INDIAN_STATE_CODES and (clean[4].isdigit() or clean[4] in LETTER_TO_DIGIT):
                clean = clean[2:]

        if len(clean) < 4:
            return clean, 0.2

        # 3. State Code Resolution (Chars 0 and 1)
        raw_state = clean[:2]
        resolved_state = self.ambiguity_resolver.resolve_state_ambiguity(raw_state)
        if resolved_state not in INDIAN_STATE_CODES:
            c0 = self.ambiguity_resolver.resolve_ambiguity(clean[0], "letter")
            c1 = self.ambiguity_resolver.resolve_ambiguity(clean[1], "letter")
            cand = c0 + c1
            if cand in INDIAN_STATE_CODES:
                resolved_state = cand
            else:
                resolved_state = self.ambiguity_resolver.resolve_state_ambiguity(cand)

        rest = clean[2:]
        if len(rest) < 3:
            corrected = resolved_state + rest
            return corrected, 0.4

        # 4. Syntactic Structural Parsing of District, Series, and Number
        if rest.isdigit():
            # If rest is 7 digits (e.g. 4705851 in MH4705851): [2 dist][1 series letter mis-OCR'd as digit][4 number]
            if len(rest) == 7:
                dist_str = rest[:2]
                series_str = self.ambiguity_resolver.resolve_ambiguity(rest[2], "letter")
                num_str = rest[3:]
                corrected = resolved_state + dist_str + series_str + num_str
            # If rest is 8 digits (e.g. 01817906 in MH01817906): [2 dist][2 series letters mis-OCR'd as digits][4 number]
            elif len(rest) == 8:
                dist_str = rest[:2]
                s0 = self.ambiguity_resolver.resolve_ambiguity(rest[2], "letter")
                s1 = self.ambiguity_resolver.resolve_ambiguity(rest[3], "letter")
                series_str = self.ambiguity_resolver.validate_rto_series(s0 + s1)
                num_str = rest[4:]
                corrected = resolved_state + dist_str + series_str + num_str
            else:
                # Vintage plate (e.g. AR048395, ML087084)
                corrected = resolved_state + rest
        else:
            # Standard plate with [District][Series][Number]
            # Backward scan for registration number digits (up to 4 digits)
            num_chars = []
            i = len(rest) - 1
            while i >= 0 and len(num_chars) < 4:
                c = rest[i]
                if c.isdigit():
                    num_chars.append(c)
                    i -= 1
                elif len(num_chars) < 2 and c in ("O", "D", "I", "Z", "S", "B"):
                    num_chars.append(self.ambiguity_resolver.resolve_ambiguity(c, "digit"))
                    i -= 1
                else:
                    break

            num_str = "".join(reversed(num_chars))
            prefix = rest[:i + 1]

            if not prefix:
                corrected = resolved_state + num_str
            else:
                # Disambiguate District (start of prefix) vs Series (rest of prefix)
                if resolved_state == "DL" and prefix[0].isdigit() and (len(prefix) > 1 and (prefix[1].isalpha() or prefix[1] in ("C", "S"))):
                    dist_str = prefix[0]
                    series_str = prefix[1:]
                elif len(prefix) >= 2 and prefix[0].isdigit() and prefix[1].isdigit():
                    dist_str = prefix[:2]
                    series_str = prefix[2:]
                elif len(prefix) >= 2 and (prefix[0].isdigit() or prefix[0] in LETTER_TO_DIGIT) and (prefix[1].isdigit() or prefix[1] in LETTER_TO_DIGIT) and not (len(prefix) == 2 and prefix.isalpha()):
                    dist_str = self.ambiguity_resolver.resolve_ambiguity(prefix[0], "digit") + self.ambiguity_resolver.resolve_ambiguity(prefix[1], "digit")
                    series_str = prefix[2:]
                elif len(prefix) >= 1 and prefix[0].isdigit():
                    dist_str = prefix[0]
                    series_str = prefix[1:]
                else:
                    dist_str = self.ambiguity_resolver.resolve_ambiguity(prefix[0], "digit")
                    series_str = prefix[1:]

                fixed_series = "".join(self.ambiguity_resolver.resolve_ambiguity(c, "letter") for c in series_str)
                fixed_num = "".join(self.ambiguity_resolver.resolve_ambiguity(c, "digit") for c in num_str)

                # Validate series against MoRTH norms
                fixed_series = self.ambiguity_resolver.validate_rto_series(fixed_series)

                corrected = resolved_state + dist_str + fixed_series + fixed_num

        # 5. Grammar Scoring
        score = 0.0
        state_prefix = corrected[:2]
        if state_prefix in INDIAN_STATE_CODES:
            score += 0.35
        elif state_prefix.isalpha():
            score += 0.15

        # Check district code (chars 2, 3)
        if len(corrected) >= 4 and corrected[2:4].isdigit():
            score += 0.25
        elif len(corrected) >= 3 and corrected[2].isdigit():
            score += 0.10

        # Check registration digits at the end
        if len(corrected) >= 8 and corrected[-4:].isdigit():
            score += 0.30
        elif len(corrected) >= 5 and corrected[-1].isdigit():
            score += 0.15

        # Full regex match bonus
        if re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$", corrected):
            score = max(score, 0.99)
        elif re.match(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$", corrected):
            score = max(score, 0.99)
        elif re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$", corrected):
            score = max(score, 0.90)

        return corrected, round(min(score, 1.0), 2)


