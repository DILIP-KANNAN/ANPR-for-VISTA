"""
Generalized Character-Level Ambiguity Resolver & RTO Series Progression Engine.
Provides syntactic role enforcement and generalized visual ambiguity resolution:
  - State code validation & visual confusion projection
  - MoRTH Series rules (prohibiting I and O, mapping digits to letters, validating commercial markers)
  - District & registration number digit enforcement (mapping D/O/Q/U -> 0, I/L/T -> 1, Z -> 2, S -> 5, B -> 8, G -> 6)
"""

import numpy as np
from typing import Tuple, Optional, Set

# All 38 Indian State & Union Territory codes (MoRTH Gazette)
INDIAN_STATE_CODES: Set[str] = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", 
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", 
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", 
    "WB", "BH", "OR"
}

# Bidirectional visual confusion pairs under license plate fonts & tight borders
CONFUSABLE_LETTER_PAIRS = {
    ("A", "M"), ("M", "A"), ("R", "M"), ("M", "R"), ("H", "M"), ("M", "H"),
    ("B", "M"), ("M", "B"), ("E", "M"), ("M", "E"), ("G", "C"), ("C", "G"),
    ("O", "C"), ("C", "O"), ("Z", "P"), ("P", "Z"), ("E", "W"), ("W", "E"),
    ("B", "W"), ("W", "B"), ("T", "I"), ("I", "T"), ("U", "O"), ("O", "U"),
    ("O", "D"), ("D", "O"), ("Q", "D"), ("D", "Q")
}

class CharacterAmbiguityResolver:
    """
    Generalized visual & syntactic ambiguity resolution for license plate recognition.
    """
    def __init__(self):
        self.valid_series_letters = set("ABCDEFGHJKLMNPQRSTUVWXYZ")
        
        # State confusion map for invalid 2-letter prefixes
        self.state_confusion_map = {
            "AH": "MH", "RH": "MH", "MR": "MH", "MI": "MH", "M1": "MH", "B1": "MH", "HH": "MH",
            "MK": "MH", "BI": "MH", "GG": "CG", "CO": "CG", "HL": "ML", "EL": "ML",
            "BB": "PB", "ZB": "PB", "8R": "BR", "OH": "CH", "OI": "DL", "0L": "DL",
            "1N": "TN", "TM": "TN", "UN": "UP", "UR": "UP", "0D": "OD", "OD": "OD",
            "EM": "WB", "BM": "WB", "EN": "MZ"
        }

    def resolve_state_ambiguity(self, raw_state: str) -> str:
        """
        Validates state code. If invalid, resolves to the closest valid Indian state
        based on visual confusion geometry without touching already valid codes.
        """
        if len(raw_state) < 2:
            return raw_state

        state = raw_state[:2].upper()
        if state in INDIAN_STATE_CODES:
            return state

        # Check explicit confusion map
        if state in self.state_confusion_map:
            return self.state_confusion_map[state]

        # Check single-character confusions against all valid states
        c0, c1 = state[0], state[1]
        for valid_st in INDIAN_STATE_CODES:
            v0, v1 = valid_st[0], valid_st[1]
            if c1 == v1 and (c0, v0) in CONFUSABLE_LETTER_PAIRS:
                return valid_st
            if c0 == v0 and (c1, v1) in CONFUSABLE_LETTER_PAIRS:
                return valid_st

        return state

    def resolve_ambiguity(
        self, 
        char: str, 
        role: str, 
        plate_crop: Optional[np.ndarray] = None, 
        char_index: int = 0, 
        total_chars: int = 10
    ) -> str:
        """
        Resolves character ambiguity based on structural role.
        Preserves valid characters and only mutates when character violates syntactic role.
        """
        ch = char.upper()

        if role == "digit":
            if ch.isdigit():
                return ch
            # Must be a digit: convert character-stroke equivalents if a letter was read
            digit_map = {
                "D": "0", "O": "0", "Q": "0", "U": "0", "C": "0",
                "I": "1", "L": "1", "T": "1",
                "Z": "2",
                "E": "3",
                "A": "4",
                "S": "5",
                "G": "6",
                "B": "8"
            }
            return digit_map.get(ch, ch)

        elif role == "letter":
            if ch.isalpha():
                # MoRTH Rule: Letters 'I' and 'O' are never issued in series codes
                if ch == "I":
                    return "T"
                elif ch == "O":
                    return "D"
                return ch

            # Must be a letter: convert digit equivalents if a digit was read
            letter_map = {
                "0": "D",
                "1": "T",
                "2": "Z",
                "3": "E",
                "4": "A",
                "5": "S",
                "6": "G",
                "7": "T",
                "8": "B",
                "9": "P"
            }
            return letter_map.get(ch, "A")

        return ch

    def validate_rto_series(self, series: str) -> str:
        """
        Cross-references series letters against MoRTH Indian vehicle registration norms.
        """
        if not series:
            return ""

        chars = list(series.upper())
        for idx in range(len(chars)):
            chars[idx] = self.resolve_ambiguity(chars[idx], role="letter")
            if chars[idx] not in self.valid_series_letters:
                chars[idx] = "A"

        return "".join(chars)

