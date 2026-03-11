"""Community card (board) detection — Phase 5.

Uses the same MobileNetV2 NN as hero card detection.
Board cards are face-up playing cards with the same design as hole cards,
so the same model applies.

Pipeline per board slot:
  1. Variance gate — if std < threshold, slot is empty → None
  2. Crop the slot region
  3. Run through NN classifier (rank + suit heads)
  4. Return card string if confidence >= threshold, else None
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from core.config import TrackerConfig, crop_region
from detection.card_detector import _classify_with_confidence
from detection.street_detector import BOARD_VARIANCE_THRESHOLD

# Minimum joint rank×suit confidence to accept a board card read.
# Lower than hero threshold (0.30) because board cards are cleaner / larger.
_BOARD_CONFIDENCE_THRESHOLD = 0.20


def detect_board_cards(
    img: Image.Image,
    cfg: TrackerConfig,
) -> List[Optional[str]]:
    """Read all board slots and return a list of card strings.

    Returns a list of up to 5 elements.  Each element is either a card string
    like "Ah" or None (slot empty or confidence too low).
    Slots are ordered left-to-right: [flop1, flop2, flop3, turn, river].
    """
    if not cfg.regions.board:
        return []

    model_path = str(Path(__file__).parent.parent / cfg.model_path)
    results: List[Optional[str]] = []

    for region in cfg.regions.board:
        variance = _slot_variance(img, region)
        if variance < BOARD_VARIANCE_THRESHOLD:
            results.append(None)
            continue

        crop = crop_region(img, region)
        try:
            card_str, conf = _classify_with_confidence(crop, model_path)
            results.append(card_str if conf >= _BOARD_CONFIDENCE_THRESHOLD else None)
        except Exception as e:
            print(f"[board_detector] NN error on board slot: {e}", file=sys.stderr)
            results.append(None)

    return results


def board_cards_to_str(cards: List[Optional[str]]) -> str:
    """Format board card list for display.  None slots shown as '?'."""
    return " ".join(c if c is not None else "?" for c in cards)


# ── helpers ───────────────────────────────────────────────────────────────────

def _slot_variance(img: Image.Image, region: list) -> float:
    crop = crop_region(img, region)
    return float(np.array(crop.convert("L"), dtype=float).std())
