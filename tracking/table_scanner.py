"""TableScanner: snapshot of full table state at hand start.

Orchestrates dealer detection, hero OCR, opponent seat OCR,
and position assignment into a single TableState object.
"""
import time
from typing import Optional

from PIL import Image

from core.config import TrackerConfig, crop_region
from detection.dealer_detector import DealerDetector
from detection.ocr import read_number, read_text
from core.table_state import PlayerState, SeatStatus, TableState, assign_positions


class TableScanner:
    """Produce a TableState snapshot from a single screenshot.

    Args:
        cfg:  TrackerConfig (regions + thresholds).
        room: Room identifier for dealer template, e.g. "PS".
    """

    def __init__(self, cfg: TrackerConfig, room: str = "PS"):
        self._cfg    = cfg
        self._dealer = DealerDetector(room, threshold=cfg.dealer_match_threshold)

    # ── public API ────────────────────────────────────────────────────────────

    def scan(self, img: Image.Image, hand_id: str) -> TableState:
        """Analyse one screenshot and return a TableState.

        This is a pure snapshot — call once at HAND_START.
        """
        cfg = self._cfg
        r   = cfg.regions

        # 1. Dealer button seat (0 = hero, 1-N = opponent, None = unknown)
        dealer_seat = self._dealer.find_dealer(img, cfg)

        # 2. Hero username + stack
        hero_username = (
            read_text(crop_region(img, r.hero_name)) if r.hero_name else None
        )
        hero_stack = (
            read_number(crop_region(img, r.hero_stack)) if r.hero_stack else None
        )

        # 3. Build player list — hero is always seat 0
        players: list[PlayerState] = [
            PlayerState(
                seat     = 0,
                username = hero_username or None,
                stack    = hero_stack,
                status   = "ACTIVE",   # hero is always active in a hand
                is_hero  = True,
            )
        ]

        # 4. Scan each opponent seat
        for i, seat_region in enumerate(r.seats):
            seat_num = i + 1
            username_raw = (
                read_text(crop_region(img, seat_region["name"]))
                if "name" in seat_region else ""
            )
            stack_raw = (
                read_text(crop_region(img, seat_region["stack"]))
                if "stack" in seat_region else ""
            )
            status = _determine_status(username_raw, stack_raw)
            stack  = (
                _parse_stack_from_raw(stack_raw) if status == "ACTIVE" else None
            )
            players.append(PlayerState(
                seat     = seat_num,
                username = username_raw.strip() or None,
                stack    = stack,
                status   = status,
                is_hero  = False,
            ))

        # 5. Assign positions
        pos_map = assign_positions(players, dealer_seat, r)
        for p in players:
            p.position = pos_map.get(p.seat)

        hero_position = next(
            (p.position for p in players if p.is_hero), None
        )

        # 6. Infer BB/SB from blind posts if possible (placeholder — Phase 4)
        big_blind   = None
        small_blind = None

        return TableState(
            dealer_seat   = dealer_seat,
            hero_seat     = 0,
            hero_username = hero_username or None,
            hero_position = hero_position,
            big_blind     = big_blind,
            small_blind   = small_blind,
            players       = players,
            timestamp     = int(time.time()),
            hand_id       = hand_id,
        )


# ── seat status helpers ───────────────────────────────────────────────────────

def _determine_status(username_raw: str, stack_raw: str) -> SeatStatus:
    """Classify a seat as ACTIVE, SITTING_OUT, or EMPTY from OCR text.

    Rules (matches PokerStars UI text):
      • username ≈ "empty" AND stack ≈ "seat"  → EMPTY
      • stack contains "sitting" or "sit out"   → SITTING_OUT
      • username is blank                        → EMPTY (no one registered)
      • otherwise                               → ACTIVE
    """
    u = username_raw.lower().strip()
    s = stack_raw.lower().strip()

    if "empty" in u and "seat" in s:
        return "EMPTY"
    if "sitting" in s or "sit out" in s or "sitout" in s:
        return "SITTING_OUT"
    if not u:
        return "EMPTY"
    return "ACTIVE"


def _parse_stack_from_raw(raw: str) -> Optional[float]:
    """Parse a stack amount from already-OCR'd text (avoids double-OCR)."""
    from detection.ocr import _parse_number
    return _parse_number(raw)
