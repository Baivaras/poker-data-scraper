"""Community card (board) detection — Phase 5.

Board slot regions in this project are calibrated tightly to the card face,
so the full-card model (card_detector_nn.pth) is used directly — the same
model and pipeline as hero card detection.

Pipeline per board slot:
  1. Variance gate  — if std < threshold, slot is empty → None
  2. Hash check     — skip NN if slot pixels are unchanged from last frame
  3. Classify       — full-card model at 128×192, confidence threshold applied
  4. Return card string if confidence >= threshold, else None
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from core.config import TrackerConfig, crop_region
from detection.card_detector import _classify_with_confidence, _img_hash
from detection.street_detector import BOARD_VARIANCE_THRESHOLD

_FULL_MODEL = str(Path(__file__).parent.parent / "NN" / "card_detector_nn.pth")

_BOARD_CONFIDENCE_THRESHOLD = 0.25

# Per-slot cache: slot_index → (image_hash, card_str)
_board_cache: Dict[int, Tuple[bytes, str]] = {}


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

    results: List[Optional[str]] = []

    for i, region in enumerate(cfg.regions.board):
        if _slot_variance(img, region) < BOARD_VARIANCE_THRESHOLD:
            results.append(None)
            continue

        slot_crop = crop_region(img, region)
        h         = _img_hash(slot_crop)

        cached = _board_cache.get(i)
        if cached is not None and cached[0] == h:
            results.append(cached[1])
            continue

        try:
            card_str, conf = _classify_with_confidence(slot_crop, _FULL_MODEL)
            result = card_str if conf >= _BOARD_CONFIDENCE_THRESHOLD else None
            if result:
                _board_cache[i] = (h, result)
            results.append(result)
        except Exception as e:
            print(f"[board_detector] NN error on slot {i}: {e}", file=sys.stderr)
            results.append(None)

    return results


def board_cards_to_str(cards: List[Optional[str]]) -> str:
    """Format board card list for display.  None slots shown as '?'."""
    return " ".join(c if c is not None else "?" for c in cards)


# ── helpers ───────────────────────────────────────────────────────────────────


def _slot_variance(img: Image.Image, region: list) -> float:
    crop = crop_region(img, region)
    return float(np.array(crop.convert("L"), dtype=float).std())
