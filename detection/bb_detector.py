"""BB/stake detection — Phase 4.

Two strategies tried in order:
1. Window title parsing  — instant, no OCR, reliable when PS title is available.
2. Pot OCR fallback      — pot after blinds = 1.5 × BB, so BB ≈ pot / 1.5.

Room-agnostic: both strategies work regardless of the poker room as long as
the title follows a "SB/BB" format or the pot region is calibrated.
"""
import re
from typing import Optional

from PIL import Image

from core.config import TrackerConfig, crop_region
from detection.ocr import read_number


def parse_bb_from_title(title: str) -> Optional[float]:
    """Extract BB amount from a PokerStars (or similar) window title.

    Recognised patterns:
      "Table 'X' - No Limit Hold'em - $0.50/$1.00"
      "Table 'X' - NL Hold'em - 50/100"
      "Zoom Poker - $1/$2 No Limit Hold'em"

    Returns the BB (the larger of the two numbers) or None.
    """
    if not title:
        return None
    # Match optional currency symbol + number / number
    pattern = r'[\$€£]?(\d+(?:\.\d+)?)\s*/\s*[\$€£]?(\d+(?:\.\d+)?)'
    match = re.search(pattern, title)
    if match:
        a = float(match.group(1))
        b = float(match.group(2))
        if a > 0 and b > 0:
            return max(a, b)   # BB is always the larger value
    return None


def detect_bb_from_pot(img: Image.Image, cfg: TrackerConfig) -> Optional[float]:
    """Estimate BB from pot region right after blinds post.

    Standard NL Hold'em: pot after both blinds = SB + BB = 1.5 × BB.
    So BB ≈ pot / 1.5.

    Only works reliably when called on the first frame after blinds finish
    posting. Returns None if OCR fails or the value is implausible.
    """
    if not cfg.regions.pot:
        return None
    pot = read_number(crop_region(img, cfg.regions.pot))
    if pot is None or pot <= 0:
        return None
    bb = round(pot / 1.5, 2)
    # Sanity gate: BB should be between 0.01 and 100 000
    if 0.01 <= bb <= 100_000:
        return bb
    return None


def detect_bb(
    img: Image.Image,
    cfg: TrackerConfig,
    window_title: Optional[str] = None,
) -> Optional[float]:
    """Detect BB using the best available strategy.

    Tries window title first (free), then pot OCR fallback.
    Returns the BB chip amount or None if detection fails.
    """
    bb = parse_bb_from_title(window_title or "")
    if bb is not None:
        return bb
    return detect_bb_from_pot(img, cfg)
