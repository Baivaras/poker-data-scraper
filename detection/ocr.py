"""Tesseract OCR helpers — pure functions, no global state."""
import re
import sys
from typing import Optional

import numpy as np
from PIL import Image

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Warning: pytesseract not installed — OCR will return empty strings.", file=sys.stderr)


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img: Image.Image, upscale: int = 3) -> Image.Image:
    """Prepare a crop for Tesseract.

    Steps:
      1. Convert to grayscale.
      2. Upscale (Tesseract works better on larger text).
      3. Invert if background is dark (PokerStars uses light text on dark bg).
      4. Binarise with Otsu-style threshold.
    """
    gray = img.convert("L")
    if upscale > 1:
        gray = gray.resize(
            (gray.width * upscale, gray.height * upscale), Image.LANCZOS
        )
    arr = np.array(gray, dtype=np.uint8)

    # Invert dark backgrounds so text becomes black on white
    if arr.mean() < 128:
        arr = 255 - arr

    # Simple binary threshold at mid-point between min and max
    lo, hi = int(arr.min()), int(arr.max())
    thresh = lo + (hi - lo) // 2
    arr = np.where(arr >= thresh, 255, 0).astype(np.uint8)

    return Image.fromarray(arr)


# ── OCR calls ─────────────────────────────────────────────────────────────────

def read_text(img: Image.Image) -> str:
    """OCR a single-line text crop → cleaned string.

    Returns empty string on failure or if Tesseract is unavailable.
    """
    if not HAS_TESSERACT:
        return ""
    try:
        proc = preprocess(img)
        raw  = pytesseract.image_to_string(
            proc, config="--psm 7 --oem 3"
        )
        return raw.strip()
    except Exception as e:
        print(f"[ocr] read_text error: {e}", file=sys.stderr)
        return ""


def read_number(img: Image.Image) -> Optional[float]:
    """OCR a numeric crop → float.

    Handles:
      • Dollar signs:   "$1,234"  → 1234.0
      • Commas:         "1,234"   → 1234.0
      • K suffix:       "1.2k"    → 1200.0
      • BB suffix:      "45 BB"   → 45.0   (returns raw number, caller interprets)
      • Plain floats:   "45.50"   → 45.5

    Returns None if no valid number found.
    """
    text = read_text(img)
    return _parse_number(text)


_ALL_IN_RE = re.compile(r'all\s*[_\-]?\s*in', re.IGNORECASE)


def read_stack(img: Image.Image) -> Optional[float]:
    """OCR a stack region → float, treating 'All In' text as 0.0.

    Use this instead of read_number for stack regions so that a player who
    has gone all-in (stack display changes to 'All In') is detected as
    current_stack=0, giving delta = last_stack = their all-in amount.
    """
    text = read_text(img)
    if _ALL_IN_RE.search(text):
        return 0.0
    return _parse_number(text)


def _parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    clean = text.strip().upper().replace(",", "").replace("$", "").replace(" ", "")

    # K suffix: "1.2K" → 1200
    m = re.search(r"([\d.]+)K", clean)
    if m:
        try:
            return float(m.group(1)) * 1000
        except ValueError:
            pass

    # Plain number (possibly with BB / other suffixes after)
    m = re.search(r"[\d.]+", clean)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass

    return None
