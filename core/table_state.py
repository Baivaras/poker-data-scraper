"""Table state dataclasses and position assignment logic."""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

SeatStatus = Literal["ACTIVE", "SITTING_OUT", "EMPTY"]

# Positions in clockwise deal order for each player count.
# Last entry is always BTN (the dealer seat).
_POSITION_CHARTS: Dict[int, List[str]] = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["CO",  "BTN", "SB", "BB"],
    5: ["UTG", "CO",  "BTN", "SB", "BB"],
    6: ["UTG", "MP",  "CO",  "BTN", "SB", "BB"],
}


@dataclass
class PlayerState:
    seat: int                    # 0 = hero, 1-5 = opponent seats
    username: Optional[str]
    stack: Optional[float]
    status: SeatStatus
    is_hero: bool
    position: Optional[str] = None  # BTN / SB / BB / UTG / MP / CO

    def to_dict(self) -> dict:
        return {
            "seat":     self.seat,
            "username": self.username,
            "stack":    self.stack,
            "status":   self.status,
            "is_hero":  self.is_hero,
            "position": self.position,
        }


@dataclass
class TableState:
    dealer_seat: Optional[int]      # 0 = hero has BTN, 1-5 = opponent seat, None = unknown
    hero_seat: int                  # always 0 in our internal representation
    hero_username: Optional[str]
    hero_position: Optional[str]
    big_blind: Optional[float]
    small_blind: Optional[float]
    players: List[PlayerState]      # all seats including hero (seat 0)
    timestamp: int
    hand_id: str

    def to_dict(self) -> dict:
        return {
            "dealer_seat":    self.dealer_seat,
            "hero_seat":      self.hero_seat,
            "hero_username":  self.hero_username,
            "hero_position":  self.hero_position,
            "big_blind":      self.big_blind,
            "small_blind":    self.small_blind,
            "players":        [p.to_dict() for p in self.players],
            "timestamp":      self.timestamp,
            "hand_id":        self.hand_id,
        }


# ── clockwise ordering ────────────────────────────────────────────────────────

def clockwise_seat_order(regions) -> List[int]:
    """Return seat numbers [0=hero, 1..N=opponents] sorted clockwise.

    Derives order from the center of each seat's name region.
    Seat 0 uses hero_name region; seats 1-N use seats[i-1]["name"].
    """
    positions: Dict[int, tuple] = {}

    if regions.hero_name:
        r = regions.hero_name
        positions[0] = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)

    for i, seat in enumerate(regions.seats):
        if "name" in seat:
            r = seat["name"]
            positions[i + 1] = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)

    if not positions:
        return []

    # Table center = centroid of all seat positions
    cx = sum(p[0] for p in positions.values()) / len(positions)
    cy = sum(p[1] for p in positions.values()) / len(positions)

    def _cw_angle(seat_num: int) -> float:
        px, py = positions[seat_num]
        dx, dy = px - cx, py - cy
        # Screen coords: y increases downward.
        # Clockwise from top = atan2(dx, -dy)
        angle = math.atan2(dx, -dy)
        return angle if angle >= 0 else angle + 2 * math.pi

    return sorted(positions.keys(), key=_cw_angle)


# ── position assignment ───────────────────────────────────────────────────────

def assign_positions(
    players: List[PlayerState],
    dealer_seat: Optional[int],
    regions,
) -> Dict[int, str]:
    """Return {seat_num: position_str} for all active seats.

    Args:
        players:     All PlayerState objects (seat 0 = hero).
        dealer_seat: Seat number of the dealer (0 = hero). None → can't assign.
        regions:     RegionConfig (used to derive clockwise order).

    Returns empty dict if dealer_seat is None or not enough players.
    """
    if dealer_seat is None:
        return {}

    clockwise = clockwise_seat_order(regions)
    if not clockwise:
        return {}

    # Build lookup: seat → status
    status_by_seat = {p.seat: p.status for p in players}

    # Active seats in clockwise order (EMPTY / SITTING_OUT excluded)
    active = [s for s in clockwise if status_by_seat.get(s) == "ACTIVE"]

    n = len(active)
    chart = _POSITION_CHARTS.get(n)
    if not chart or dealer_seat not in active:
        return {}

    dealer_idx   = active.index(dealer_seat)
    btn_in_chart = max(0, n - 3)  # index of BTN in chart (verified for 2-6p)

    result: Dict[int, str] = {}
    for chart_offset, pos_name in enumerate(chart):
        seat_offset = chart_offset - btn_in_chart
        seat        = active[(dealer_idx + seat_offset) % n]
        result[seat] = pos_name

    return result
