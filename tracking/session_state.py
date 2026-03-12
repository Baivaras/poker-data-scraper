"""SessionState — persists usernames across hands, accumulates hand history,
and writes JSONL session logs to session_logs/.

One SessionState instance = one table session (one PokerStars window).
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from core.config import TrackerConfig, crop_region
from detection.ocr import read_text
from detection.region_cache import RegionCache

# ── label filtering ────────────────────────────────────────────────────────────

# Known action / status labels that can appear in the name region.
# OCR reads matching these should NOT overwrite a stored username.
# Extend per-room as needed.
_LABEL_BLOCKLIST: frozenset = frozenset({
    "post bb", "post sb",
    "bet", "raise", "call", "check", "fold",
    "all in", "all-in", "allin",
    "sitting out", "sit out", "sitting-out",
    "bb", "sb",
})

# Labels that signal the seat is now empty — clear stored username.
_EMPTY_LABELS: frozenset = frozenset({
    "empty seat", "empty",
})


_CLEAN_RE = re.compile(r"['\"/\\|,.\[\](){}<>:;!?@#~`^&*+=_\-]")


def _clean(text: str) -> str:
    """Strip OCR noise characters before blocklist comparison or storage."""
    return _CLEAN_RE.sub("", text).strip()


def _is_label(text: str) -> bool:
    return _clean(text).lower() in _LABEL_BLOCKLIST


def _is_empty(text: str) -> bool:
    return _clean(text).lower() in _EMPTY_LABELS


def parse_table_name(window_title: Optional[str]) -> Optional[str]:
    """Extract table name from a PokerStars window title.

    Expected format: "PokerStars - $0.50/$1.00 - Table 'Altair' 6-max"
    Returns: "Altair", or None if not found.
    """
    if not window_title:
        return None
    m = re.search(r"Table\s+'([^']+)'", window_title, re.IGNORECASE)
    return m.group(1) if m else None


# ── SessionState ───────────────────────────────────────────────────────────────

class SessionState:
    """Owns the session-level username registry and hand log for one table session.

    Usernames are updated continuously from name-region OCR; action labels and
    OCR noise are filtered via blocklist so stored names are always real.

    Completed hands are appended as JSON lines to
    session_logs/session_{session_id}.jsonl immediately on hand completion.
    """

    def __init__(
        self,
        session_id: str,
        table_name: Optional[str] = None,
        log_dir: str = "session_logs",
    ):
        self.session_id = session_id
        self.table_name = table_name
        self._usernames: Dict[int, Optional[str]] = {}   # seat → username (0 = hero)
        self._name_cache = RegionCache()
        self._log_path   = self._open_log(log_dir, session_id, table_name)

    # ── username management ────────────────────────────────────────────────────

    def update_usernames(self, img: Image.Image, cfg: TrackerConfig) -> None:
        """Scan all name regions and update stored usernames when real text is seen.

        Skips regions whose pixels haven't changed since last call (RegionCache).
        Safe to call every frame — cost is near zero on static frames.
        """
        for i, seat_region in enumerate(cfg.regions.seats):
            if "name" not in seat_region:
                continue
            region = seat_region["name"]
            if self._name_cache.changed(img, region):
                self._read_and_store(i + 1, img, region)

        if cfg.regions.hero_name and self._name_cache.changed(img, cfg.regions.hero_name):
            self._read_and_store(0, img, cfg.regions.hero_name)

    def get_username(self, seat: int) -> Optional[str]:
        """Return the best-known username for this seat (0 = hero), or None."""
        return self._usernames.get(seat)

    def clear_seat(self, seat: int) -> None:
        """Explicitly clear a seat's username (player left the table)."""
        self._usernames[seat] = None

    # ── hand recording ─────────────────────────────────────────────────────────

    def record_hand(self, hand_state) -> None:
        """Append a completed HandState as one JSON line to the session log."""
        if self._log_path is None:
            return
        record = _hand_to_dict(hand_state)
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            import sys
            print(f"[SessionState] Failed to write hand log: {e}", file=sys.stderr)

    # ── internals ─────────────────────────────────────────────────────────────

    def _read_and_store(self, seat: int, img: Image.Image, region) -> None:
        raw  = read_text(crop_region(img, region)).strip()
        text = _clean(raw)
        if not text:
            return
        if _is_empty(text):
            if self._usernames.get(seat) is not None:
                self._usernames[seat] = None
        elif not _is_label(text) and len(text) >= 2:
            if self._usernames.get(seat) != text:
                self._usernames[seat] = text

    @staticmethod
    def _open_log(
        log_dir: str, session_id: str, table_name: Optional[str]
    ) -> Optional[Path]:
        import sys
        try:
            d = Path(log_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"session_{session_id}.jsonl"
            header = {
                "type":       "session",
                "session_id": session_id,
                "table":      table_name,
                "started_at": int(time.time()),
            }
            with path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(header) + "\n")
            return path
        except Exception as e:
            print(f"[SessionState] Could not create log: {e}", file=sys.stderr)
            return None


# ── serialisation ──────────────────────────────────────────────────────────────

def _hand_to_dict(hs) -> dict:
    positions = {}
    if hs.table_state:
        for p in hs.table_state.players:
            key = "hero" if p.seat == 0 else f"seat{p.seat}"
            positions[key] = p.position

    return {
        "type":            "hand",
        "hand_id":         hs.hand_id,
        "started_at":      hs.started_at,
        "completed_at":    hs.completed_at,
        "bb_amount":       hs.bb_amount,
        "hero_cards":      [c.to_str() for c in hs.hero_cards] if hs.hero_cards else None,
        "community_cards": list(hs.community_cards),
        "positions":       positions,
        "hero_folded":     hs.hero_folded,
        "actions": [
            {
                "seat":                 a.seat,
                "username":             a.username,
                "street":               a.street,
                "action":               a.action,
                "amount":               a.amount,
                "amount_bb":            a.amount_bb,
                "amount_dollars":       a.amount_dollars,
                "street_total":         a.street_total,
                "street_total_bb":      a.street_total_bb,
                "street_total_dollars": a.street_total_dollars,
                "stack_before":         a.stack_before,
                "stack_after":          a.stack_after,
                "timestamp":            a.timestamp,
            }
            for a in hs.all_actions
        ],
    }
