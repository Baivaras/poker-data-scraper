"""Dealer button detector via template matching.

Checks each labeled dealer region (hero + all opponent seats) and returns
which seat has the best match above the confidence threshold.

Room-agnostic: pass room="PS" to load images/PS_dealer_button.png.
Adding a new room = drop images/GGPoker_dealer_button.png and pass room="GGPoker".
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.config import TrackerConfig, crop_region


class DealerDetector:
    """Template-match the dealer chip across all labeled seat dealer regions.

    Args:
        room:      Room identifier — loads images/{room}_dealer_button.png.
        threshold: Minimum NCC score (0-1) to accept a match.
    """

    def __init__(self, room: str = "PS", threshold: Optional[float] = None):
        self._room      = room
        self._threshold = threshold  # None → use cfg.dealer_match_threshold at call time
        self._template  = self._load_template(room)

    # ── public API ────────────────────────────────────────────────────────────

    def find_dealer(self, img: Image.Image, cfg: TrackerConfig) -> Optional[int]:
        """Return the seat number that holds the dealer button, or None.

        Return values:
            0       → hero has the dealer button
            1-N     → opponent seat N has the button (1-based index)
            None    → no match above threshold found

        Checks every region that has a "dealer" key in regions.json, plus
        hero_dealer if configured.
        """
        if self._template is None:
            return None

        threshold = self._threshold if self._threshold is not None \
                    else cfg.dealer_match_threshold
        scales    = cfg.dealer_scales

        best_score = -1.0
        best_seat  = None

        # Hero dealer region (seat 0)
        if cfg.regions.hero_dealer:
            score = self._match_region(img, cfg.regions.hero_dealer, scales)
            if score > best_score:
                best_score = score
                best_seat  = 0

        # Opponent seat dealer regions (seats 1-N)
        for i, seat in enumerate(cfg.regions.seats):
            if "dealer" not in seat:
                continue
            score = self._match_region(img, seat["dealer"], scales)
            if score > best_score:
                best_score = score
                best_seat  = i + 1  # 1-based

        if best_score >= threshold:
            return best_seat

        return None

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_template(room: str) -> Optional[np.ndarray]:
        path = Path(__file__).parent.parent / "images" / f"{room}_dealer_button.png"
        if not path.exists():
            print(
                f"[DealerDetector] Template not found: {path}\n"
                f"  Drop a cropped PNG at images/{room}_dealer_button.png to enable detection.",
                file=sys.stderr,
            )
            return None
        img  = Image.open(path).convert("L")
        arr  = np.array(img, dtype=np.float32)
        arr -= arr.mean()
        std  = arr.std()
        if std > 0:
            arr /= std
        return arr

    def _match_region(
        self,
        img: Image.Image,
        region: list,
        scales: list,
    ) -> float:
        """Crop the region, try each template scale, return best NCC score."""
        crop = crop_region(img, region).convert("L")
        cw, ch = crop.size
        if cw < 4 or ch < 4:
            return -1.0

        crop_arr = np.array(crop, dtype=np.float32)
        best = -1.0

        for scale in scales:
            tw = max(2, int(self._template.shape[1] * scale))
            th = max(2, int(self._template.shape[0] * scale))

            # Resize template to this scale
            t_img    = Image.fromarray(
                ((self._template - self._template.min()) /
                 (self._template.max() - self._template.min() + 1e-6) * 255
                 ).astype(np.uint8)
            ).resize((tw, th), Image.LANCZOS)
            t_arr    = np.array(t_img, dtype=np.float32)

            # Resize crop to match template dimensions for direct NCC
            c_resized = np.array(
                crop.resize((tw, th), Image.LANCZOS), dtype=np.float32
            )

            score = _ncc(t_arr, c_resized)
            if score > best:
                best = score

        return best


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation between two same-shape arrays. Range [-1, 1]."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(a * b) / denom)
