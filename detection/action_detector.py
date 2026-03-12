"""Player action detection via stack delta + bet chip cross-reference.

Primary signal:  stack decreased → betting action
Secondary signal: bet chip region OCR → confirms amount
Fold signal:     cards region variance drops → FOLD

Never false-positives on None stacks (OCR failure).
Requires delta to be stable across 2 consecutive frames before emitting.
"""
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from PIL import Image

from core.config import TrackerConfig, crop_region
from core.hand_state import ActionType, PlayerAction, Street
from detection.ocr import read_number, read_stack
from detection.region_cache import RegionCache

# Fraction of stack that can change due to OCR noise without triggering an action.
# e.g. 0.05 = ignore deltas < 5% of last known stack (covers minor misreads).
_NOISE_FRACTION = 0.005   # 0.5% — keeps noise floor below 1 BB for stacks up to 100 BB
_MIN_NOISE_CHIPS = 0.5    # always ignore deltas smaller than this regardless of stack size

# Board slot variance threshold (reused from street_detector)
_CARD_VARIANCE_THRESHOLD = 40.0


@dataclass
class _SeatState:
    last_stack:    Optional[float] = None
    last_bet:      Optional[float] = None
    cards_present: bool            = True   # False once fold detected
    # Pending delta (must be stable across N frames before emitting)
    pending_delta: Optional[float] = None
    pending_run:   int             = 0


class ActionDetector:
    """Detect player actions frame-by-frame via stack delta.

    Usage:
        detector = ActionDetector(cfg)
        detector.snapshot(img)                          # call once at hand start
        actions = detector.detect(img, street, call_amt)  # call every frame
        detector.reset_street_bets()                    # call on street change
    """

    DELTA_DEBOUNCE = 2   # frames a delta must be stable before emitting

    def __init__(self, cfg: TrackerConfig, debug: bool = False):
        self._cfg    = cfg
        self._debug  = debug
        self._seats: Dict[int, _SeatState] = {}
        self._current_call_amount: float   = 0.0
        self._last_actor: Optional[int]    = None   # seat that last made a real action
        self._street_committed: Dict[int, float] = {}  # total chips committed per seat this street
        self._region_cache = RegionCache()   # skips fold checks when cards region unchanged

    def _dbg(self, msg: str) -> None:
        if self._debug:
            print(f"\033[2m  [action] {msg}\033[0m", file=sys.stderr, flush=True)

    # ── public API ────────────────────────────────────────────────────────────

    def snapshot(self, img: Image.Image) -> None:
        """Seed last-known stacks for all seats + hero. Call once at hand start."""
        self._seats = {}
        cfg = self._cfg

        # Hero (seat 0)
        hero_stack = None
        if cfg.regions.hero_stack:
            hero_stack = read_stack(crop_region(img, cfg.regions.hero_stack))
        self._seats[0] = _SeatState(last_stack=hero_stack, cards_present=True)
        self._dbg(f"snapshot  seat0(hero)  stack={hero_stack}")

        # Opponent seats (1-N)
        for i, seat_region in enumerate(cfg.regions.seats):
            seat_num = i + 1
            stack = None
            if "stack" in seat_region:
                stack = read_stack(crop_region(img, seat_region["stack"]))
            cards = True
            if "cards" in seat_region:
                cards = _has_cards(img, seat_region["cards"])
            self._seats[seat_num] = _SeatState(last_stack=stack, cards_present=cards)
            self._dbg(f"snapshot  seat{seat_num}  stack={stack}  cards={cards}")

    def detect(
        self,
        img: Image.Image,
        street: Street,
        table_state,              # TableState — for username lookup
    ) -> list[PlayerAction]:
        """Detect any new actions this frame. Returns list (usually 0 or 1 items)."""
        actions   = []
        cfg       = self._cfg
        usernames = {p.seat: p.username for p in table_state.players} \
                    if table_state else {}

        # ── opponent seats ────────────────────────────────────────────────────
        for i, seat_region in enumerate(cfg.regions.seats):
            seat_num   = i + 1
            seat_state = self._seats.get(seat_num)
            if seat_state is None:
                continue

            # Fold detection — skip variance check when cards region is unchanged
            if "cards" in seat_region:
                if not seat_state.cards_present:
                    continue   # already folded
                if self._region_cache.changed(img, seat_region["cards"]):
                    if not _has_cards(img, seat_region["cards"]):
                        seat_state.cards_present = False
                        actions.extend(self._infer_checks(seat_num, street, usernames))
                        actions.append(PlayerAction(
                            seat      = seat_num,
                            username  = usernames.get(seat_num),
                            action    = "FOLD",
                            amount    = None,
                            amount_bb = None,
                            street    = street,
                            timestamp = _now(),
                        ))
                        self._last_actor = seat_num
                        continue
            elif not seat_state.cards_present:
                continue   # already folded, no cards region to re-check

            if "stack" not in seat_region:
                continue

            current_stack = read_stack(crop_region(img, seat_region["stack"]))
            self._dbg(f"detect  seat{seat_num}  ocr={current_stack}  last={seat_state.last_stack}")
            action = self._check_delta(
                seat_num, seat_state, current_stack,
                seat_region.get("bet"), img,
                street, usernames.get(seat_num),
            )
            if action:
                actions.extend(self._infer_checks(seat_num, street, usernames))
                actions.append(action)
                self._last_actor = seat_num

        # ── hero stack refresh ────────────────────────────────────────────────
        if cfg.regions.hero_stack:
            current = read_stack(crop_region(img, cfg.regions.hero_stack))
            if current is not None:
                self._seats[0].last_stack = current

        return actions

    def detect_hero_action(
        self,
        stack_before: Optional[float],
        img: Image.Image,
        street: Street,
        username: Optional[str],
    ) -> Optional[PlayerAction]:
        """Classify hero's action from stack delta after buttons disappear."""
        if stack_before is None or self._cfg.regions.hero_stack is None:
            return None

        current = read_stack(crop_region(img, self._cfg.regions.hero_stack))
        if current is None:
            return None

        delta  = stack_before - current
        action = self._classify(delta, current, self._current_call_amount)
        if action is None:
            return None

        if action not in ("FOLD", "CHECK"):
            self._update_call_amount(delta)
            self._street_committed[0] = self._street_committed.get(0, 0.0) + delta

        street_total = self._street_committed.get(0, 0.0) or None

        self._seats[0].last_stack = current
        return PlayerAction(
            seat         = 0,
            username     = username,
            action       = action,
            amount       = delta if action not in ("FOLD", "CHECK") else None,
            amount_bb    = None,
            street       = street,
            timestamp    = _now(),
            street_total = street_total,
            stack_before = stack_before,
            stack_after  = current,
        )

    def reset_street_bets(self) -> None:
        """Call on every street change — resets current call amount and action order."""
        self._current_call_amount = 0.0
        self._last_actor          = None
        self._street_committed    = {}

    def hero_stack_now(self, img: Image.Image) -> Optional[float]:
        """Read hero's current stack (for snapshotting before hero acts)."""
        if self._cfg.regions.hero_stack is None:
            return None
        return read_stack(crop_region(img, self._cfg.regions.hero_stack))

    def reset(self) -> None:
        self._seats                = {}
        self._current_call_amount  = 0.0
        self._last_actor           = None
        self._street_committed     = {}
        self._region_cache.reset()

    # ── internals ─────────────────────────────────────────────────────────────

    def record_actor(self, seat: int) -> None:
        """Record that a seat just acted (call after hero action is detected)."""
        self._last_actor = seat

    def _infer_checks(
        self,
        current_actor: int,
        street:        Street,
        usernames:     dict,
    ) -> list[PlayerAction]:
        """Return implied CHECK actions for seats skipped between _last_actor and current_actor.

        Includes seat 0 (hero) in the position ordering so that actions after
        hero's turn correctly infer CHECKs for skipped opponents.  Hero itself
        is never emitted as a CHECK here — hero actions are handled separately.
        """
        last = self._last_actor
        if last is None or last == current_actor:
            return []

        # All active seats including hero (seat 0) for position ordering.
        # Seat 0 anchors the ordering but is never emitted as an inferred CHECK.
        ordered = sorted(
            s for s, st in self._seats.items() if st.cards_present
        )
        if len(ordered) < 2:
            return []
        if last not in ordered or current_actor not in ordered:
            return []

        last_i = ordered.index(last)
        curr_i = ordered.index(current_actor)
        if last_i == curr_i:
            return []

        checks = []
        i = (last_i + 1) % len(ordered)
        while i != curr_i:
            seat = ordered[i]
            if seat != 0:   # hero's checks are handled externally
                self._dbg(f"  infer CHECK  seat{seat}  (between seat{last} and seat{current_actor})")
                checks.append(PlayerAction(
                    seat      = seat,
                    username  = usernames.get(seat),
                    action    = "CHECK",
                    amount    = None,
                    amount_bb = None,
                    street    = street,
                    timestamp = _now(),
                ))
            i = (i + 1) % len(ordered)

        return checks

    def _check_delta(
        self,
        seat_num:    int,
        seat_state:  _SeatState,
        current:     Optional[float],
        bet_region:  Optional[list],
        img:         Image.Image,
        street:      Street,
        username:    Optional[str],
    ) -> Optional[PlayerAction]:
        if current is None or seat_state.last_stack is None:
            if current is not None:
                seat_state.last_stack = current
            self._dbg(f"  seat{seat_num}  skip: ocr={current} last={seat_state.last_stack} (None)")
            seat_state.pending_delta = None
            seat_state.pending_run   = 0
            return None

        delta = seat_state.last_stack - current

        # Ignore noise-level changes
        noise_floor = max(_MIN_NOISE_CHIPS,
                          seat_state.last_stack * _NOISE_FRACTION)
        if abs(delta) < noise_floor:
            self._dbg(f"  seat{seat_num}  delta={delta:.0f} < noise_floor={noise_floor:.0f}, skip")
            seat_state.pending_delta = None
            seat_state.pending_run   = 0
            return None

        # Require delta to be stable across N frames
        if seat_state.pending_delta is not None and \
                abs(seat_state.pending_delta - delta) < noise_floor:
            seat_state.pending_run += 1
        else:
            seat_state.pending_delta = delta
            seat_state.pending_run   = 1

        self._dbg(f"  seat{seat_num}  delta={delta:.0f}  pending_run={seat_state.pending_run}/{self.DELTA_DEBOUNCE}")

        if seat_state.pending_run < self.DELTA_DEBOUNCE:
            return None

        # Confirmed action — classify
        action = self._classify(delta, current, self._current_call_amount)
        if action is None:
            self._dbg(f"  seat{seat_num}  classify→None (delta={delta:.0f})")
            seat_state.last_stack    = current
            seat_state.pending_delta = None
            seat_state.pending_run   = 0
            return None

        # Cross-reference bet chip for amount (secondary signal)
        confirmed_amount: Optional[float] = delta if delta > 0 else None
        if bet_region and delta > 0:
            chip_amount = read_number(crop_region(img, bet_region))
            if chip_amount is not None and abs(chip_amount - delta) / max(delta, 1) < 0.1:
                confirmed_amount = chip_amount   # within 10% — use chip display

        if action not in ("FOLD", "CHECK"):
            self._update_call_amount(delta)
            self._street_committed[seat_num] = (
                self._street_committed.get(seat_num, 0.0) + (confirmed_amount or delta)
            )

        street_total = self._street_committed.get(seat_num, 0.0) or None

        stack_before = seat_state.last_stack
        self._dbg(f"  seat{seat_num}  ACTION={action}  amount={confirmed_amount}  "
                  f"street_total={street_total}  call_amt={self._current_call_amount:.0f}")
        seat_state.last_stack    = current
        seat_state.pending_delta = None
        seat_state.pending_run   = 0

        return PlayerAction(
            seat         = seat_num,
            username     = username,
            action       = action,
            amount       = confirmed_amount if action not in ("FOLD", "CHECK") else None,
            amount_bb    = None,
            street       = street,
            timestamp    = _now(),
            street_total = street_total,
            stack_before = stack_before,
            stack_after  = current,
        )

    def _classify(
        self,
        delta:        float,
        current:      float,
        call_amount:  float,
    ) -> Optional[ActionType]:
        if delta <= 0:
            return None          # stack went up (rebuy) or unchanged
        if current == 0:
            return "ALL_IN"
        noise = max(_MIN_NOISE_CHIPS, delta * _NOISE_FRACTION)
        if call_amount > 0 and abs(delta - call_amount) <= noise:
            return "CALL"
        if delta > call_amount:
            return "RAISE" if call_amount > 0 else "BET"
        return "CALL"            # partial call (short stack)

    def _update_call_amount(self, delta: float) -> None:
        if delta > self._current_call_amount:
            self._current_call_amount = delta


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_cards(img: Image.Image, region: list) -> bool:
    crop = crop_region(img, region)
    arr  = np.array(crop.convert("L"), dtype=float)
    return float(arr.std()) > _CARD_VARIANCE_THRESHOLD


def _now() -> int:
    import time
    return int(time.time())
