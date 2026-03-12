"""Street detection via board slot variance gate.

Checks each of the 5 board slots — if grayscale std > threshold, a card is
present. Maps card count to street:
  0 → PREFLOP
  3 → FLOP    (1/2 are animation frames — stabilised with debounce)
  4 → TURN
  5 → RIVER
"""
import numpy as np
from PIL import Image

from core.config import TrackerConfig, crop_region
from core.hand_state import Street
from detection.region_cache import RegionCache

BOARD_VARIANCE_THRESHOLD = 40.0  # same as card detector


def _slot_has_card(img: Image.Image, region: list) -> bool:
    crop = crop_region(img, region)
    arr  = np.array(crop.convert("L"), dtype=float)
    return float(arr.std()) > BOARD_VARIANCE_THRESHOLD


def count_board_cards(img: Image.Image, cfg: TrackerConfig) -> int:
    """Return the number of community card slots that contain a card (0-5)."""
    return sum(
        1 for region in cfg.regions.board
        if _slot_has_card(img, region)
    )


def card_count_to_street(count: int) -> Street:
    """Map board card count to street name.

    Counts of 1 or 2 are transient animation frames — treated as FLOP
    since the flop is dealt all at once and we'll debounce before acting.
    """
    if count == 0:
        return "PREFLOP"
    if count <= 3:
        return "FLOP"
    if count == 4:
        return "TURN"
    return "RIVER"


class StreetDetector:
    """Debounced street detector — requires N stable frames before emitting change.

    Args:
        debounce: Number of consecutive frames with the same count before
                  confirming a street change (default 2).
    """

    def __init__(self, debounce: int = 2):
        self._debounce      = debounce
        self._current_count = 0
        self._candidate     = 0
        self._candidate_run = 0
        self._region_cache  = RegionCache()
        # Last known card presence per slot index — used when region is cached
        self._slot_present: dict = {}

    @property
    def current_street(self) -> Street:
        return card_count_to_street(self._current_count)

    def update(self, img: Image.Image, cfg: TrackerConfig) -> tuple[Street, bool]:
        """Read board slots and return (street, changed).

        changed=True only on the frame a stable street transition is confirmed.
        Board slot variance checks are skipped when the slot pixels are unchanged.
        """
        count = 0
        for i, region in enumerate(cfg.regions.board):
            if self._region_cache.changed(img, region):
                present = _slot_has_card(img, region)
                self._slot_present[i] = present
            else:
                present = self._slot_present.get(i, False)
            if present:
                count += 1

        if count != self._candidate:
            self._candidate     = count
            self._candidate_run = 1
        else:
            self._candidate_run += 1

        if self._candidate_run >= self._debounce and count != self._current_count:
            self._current_count = count
            return card_count_to_street(count), True

        return card_count_to_street(self._current_count), False

    def reset(self):
        self._current_count = 0
        self._candidate     = 0
        self._candidate_run = 0
        self._region_cache.reset()
        self._slot_present.clear()
