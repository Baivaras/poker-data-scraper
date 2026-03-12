"""GameTracker — unified poller replacing HeroWatcher.

Single polling loop handles the full hand lifecycle:
  IDLE → HAND_STARTING → HAND_ACTIVE → HAND_COMPLETE → IDLE

Emits four callbacks:
  on_hand_start(hero_cards, table_state)
  on_street_change(street)
  on_player_action(action)           # fires for ALL players including after hero folds
  on_your_turn(hand_state)           # includes full context since last hero action
  on_hand_complete(hand_state)
"""
import sys
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from detection.action_detector import ActionDetector
from detection.bb_detector import detect_bb
from detection.board_detector import detect_board_cards
from detection.capture import get_pokerstars_window, screenshot
from detection.card_detector import detect_hero_cards
from core.config import TrackerConfig, crop_region, load_config
from core.hand_state import ActionRound, HandState, PlayerAction, Street
from core.models import HeroCardsResult
from detection.region_cache import RegionCache
from detection.street_detector import StreetDetector
from tracking.table_scanner import TableScanner
from core.table_state import TableState


class _State(Enum):
    IDLE           = auto()
    HAND_STARTING  = auto()
    HAND_ACTIVE    = auto()
    HAND_COMPLETE  = auto()


class GameTracker:
    """Unified game tracker — Phase 3 main entry point.

    Args:
        cfg:               TrackerConfig loaded from regions.json.
        room:              Room identifier for dealer template.
        on_hand_start:     Called when hero cards are confirmed + table scanned.
        on_street_change:  Called on stable street transition.
        on_player_action:  Called for every detected player action (all seats).
        on_your_turn:      Called when action buttons appear (hero must act).
        on_hand_complete:  Called when hand ends (fires even if hero folded).
    """

    HERO_DEBOUNCE     = 2    # frames of hero cards present to start hand
    HERO_ABSENT_LIMIT = 3    # frames of hero cards absent to end hand (if board also clear)
    BTN_SAT_THRESHOLD = 0.04 # fraction of high-sat pixels = buttons visible

    def __init__(
        self,
        cfg:               TrackerConfig,
        room:              str = "PS",
        on_hand_start:     Optional[Callable] = None,
        on_street_change:  Optional[Callable] = None,
        on_player_action:  Optional[Callable] = None,
        on_your_turn:      Optional[Callable] = None,
        on_hand_complete:  Optional[Callable] = None,
        debug:             bool = False,
    ):
        self._cfg              = cfg
        self._on_hand_start    = on_hand_start    or (lambda *a: None)
        self._on_street_change = on_street_change or (lambda *a: None)
        self._on_player_action = on_player_action or (lambda *a: None)
        self._on_your_turn     = on_your_turn     or (lambda *a: None)
        self._on_hand_complete = on_hand_complete or (lambda *a: None)

        self._scanner      = TableScanner(cfg, room)
        self._streets      = StreetDetector(debounce=2)
        self._actions      = ActionDetector(cfg, debug=debug)
        self._btn_cache    = RegionCache()   # skips saturation check when button region unchanged

        self._state:        _State           = _State.IDLE
        self._hand_state:   Optional[HandState] = None
        self._hand_counter: int              = 0
        self._running:      bool             = False

        # Debounce counters
        self._hero_present_frames = 0
        self._hero_absent_frames  = 0

        # Action round accumulation
        self._current_round: Optional[ActionRound] = None
        self._round_counter: int = 0

        # YOUR_TURN tracking
        self._btn_was_visible:   bool            = False
        self._hero_stack_before: Optional[float] = None

        # Live mode window title (for BB detection from title)
        self._window_title: Optional[str] = None

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        """Live mode: poll PokerStars window at cfg.poll_ms intervals."""
        print(f"[GameTracker] Polling every {self._cfg.poll_ms}ms", file=sys.stderr)
        self._running = True
        interval = self._cfg.poll_ms / 1000.0
        try:
            while self._running:
                t0  = time.monotonic()
                win = get_pokerstars_window()
                if win is None:
                    print("[GameTracker] PokerStars window not found — retrying...",
                          file=sys.stderr)
                else:
                    self._window_title = win.get("title", "")
                    img = screenshot(win)
                    if img:
                        self.process_frame(img)
                elapsed = time.monotonic() - t0
                sleep_s = max(0.0, interval - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[GameTracker] Stopped.", file=sys.stderr)
        finally:
            self._running = False

    def stop(self):
        self._running = False

    def process_frame(self, img: Image.Image) -> None:
        """Process a single screenshot through the state machine."""
        if self._state == _State.IDLE:
            self._state_idle(img)
        elif self._state == _State.HAND_STARTING:
            self._state_hand_starting(img)
        elif self._state == _State.HAND_ACTIVE:
            self._state_hand_active(img)

    def replay_dir(self, path: str) -> None:
        """Test mode: replay sorted PNGs through the full pipeline."""
        self._reset_state()
        p      = Path(path)
        images = sorted(
            p.glob("*.png"),
            key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
        )
        if not images:
            print(f"[GameTracker] No PNG files in {path}", file=sys.stderr)
            return
        print(f"[GameTracker] Replaying {len(images)} images from {path}",
              file=sys.stderr)
        for img_path in images:
            try:
                self.process_frame(Image.open(img_path).convert("RGB"))
            except Exception as e:
                print(f"[GameTracker] Error on {img_path.name}: {e}", file=sys.stderr)

    # ── state handlers ────────────────────────────────────────────────────────

    def _state_idle(self, img: Image.Image):
        """Wait for hero cards to appear (variance gate + debounce)."""
        result = detect_hero_cards(img, self._cfg)
        if result is not None:
            self._hero_present_frames += 1
        else:
            self._hero_present_frames = 0

        if self._hero_present_frames >= self.HERO_DEBOUNCE:
            self._hero_present_frames = 0
            self._state = _State.HAND_STARTING
            self._state_hand_starting(img)

    def _state_hand_starting(self, img: Image.Image):
        """Hero cards confirmed — run Phase 2 snapshot + initialise hand state."""
        result = detect_hero_cards(img, self._cfg)
        if result is None:
            # Cards disappeared before we could read — back to IDLE
            self._state = _State.IDLE
            return

        self._hand_counter += 1
        hand_id = f"hand_{self._hand_counter:04d}"

        # Phase 2 snapshot
        table_state = self._scanner.scan(img, hand_id)

        # Seed stacks for action detection
        self._actions.snapshot(img)

        # BB detection (Phase 4)
        bb = detect_bb(img, self._cfg, self._window_title)

        # Initialise hand state
        self._streets.reset()
        self._streets.update(img, self._cfg)   # seed current street
        self._round_counter = 0
        self._current_round = self._new_round("PREFLOP")
        self._btn_was_visible   = False
        self._hero_stack_before = None
        self._hero_absent_frames = 0

        self._hand_state = HandState(
            hand_id     = hand_id,
            hero_cards  = result.cards,
            table_state = table_state,
            bb_amount   = bb,
        )

        self._state = _State.HAND_ACTIVE
        self._on_hand_start(result, table_state)

    def _state_hand_active(self, img: Image.Image):
        hs = self._hand_state

        # ── 1. Street detection ───────────────────────────────────────────────
        street, changed = self._streets.update(img, self._cfg)
        if changed:
            hs.current_street = street
            self._close_round()
            self._actions.reset_street_bets()
            self._current_round = self._new_round(street)
            # Phase 5: read community cards when board changes
            hs.community_cards = [
                c for c in detect_board_cards(img, self._cfg) if c is not None
            ]
            self._on_street_change(street, hs.community_cards)

        # ── 2. Hand end detection ─────────────────────────────────────────────
        hero_result = detect_hero_cards(img, self._cfg)
        hero_present = hero_result is not None

        if hero_present:
            self._hero_absent_frames = 0
        else:
            self._hero_absent_frames += 1

        board_count = self._streets._current_count

        if not hero_present and not hs.hero_folded:
            if board_count == 0:
                # No cards, no board — hand is over
                if self._hero_absent_frames >= self.HERO_ABSENT_LIMIT:
                    self._end_hand()
                    return
            else:
                # Hero folded mid-hand — keep tracking
                hs.hero_folded = True

        if hs.hero_folded and board_count == 0 and \
                self._hero_absent_frames >= self.HERO_ABSENT_LIMIT:
            self._end_hand()
            return

        # ── 3. Player action detection ────────────────────────────────────────
        new_actions = self._actions.detect(img, street, hs.table_state)
        for action in new_actions:
            self._fill_bb(action)
            hs.all_actions.append(action)
            self._route_action(action)
            self._on_player_action(action)

        # ── 4. YOUR_TURN detection (action buttons, cached) ───────────────────
        if not hs.hero_folded:
            if self._btn_cache.changed(img, self._cfg.regions.action):
                btn_visible = _action_buttons_visible(img, self._cfg)
            else:
                btn_visible = self._btn_was_visible

            if btn_visible and not self._btn_was_visible:
                # Buttons just appeared → hero's turn
                self._hero_stack_before = self._actions.hero_stack_now(img)
                self._on_your_turn(hs)

            elif not btn_visible and self._btn_was_visible:
                # Buttons just disappeared → hero acted
                hero_action = self._actions.detect_hero_action(
                    self._hero_stack_before, img, street,
                    hs.table_state.hero_username if hs.table_state else None,
                )
                if hero_action:
                    if hero_action.action == "FOLD":
                        hs.hero_folded = True
                    self._fill_bb(hero_action)
                    hs.all_actions.append(hero_action)
                    self._route_action(hero_action)
                    self._on_player_action(hero_action)
                    self._actions.record_actor(0)   # seat 0 = hero
                self._hero_stack_before = None

            self._btn_was_visible = btn_visible

    # ── BB annotation ─────────────────────────────────────────────────────────

    def _fill_bb(self, action: PlayerAction) -> None:
        """Fill amount_bb, amount_dollars, street_total_bb, street_total_dollars in-place.

        The table always displays stacks/bets in BBs, so the OCR value IS the
        BB amount.  Dollar conversion uses bb_amount (dollar value of 1 BB)
        parsed from the window title.
        """
        bb = self._hand_state.bb_amount if self._hand_state else None

        if action.amount is not None:
            if action.amount_bb is None:
                action.amount_bb = action.amount
            if action.amount_dollars is None and bb:
                action.amount_dollars = round(action.amount * bb, 2)

        if action.street_total is not None:
            if action.street_total_bb is None:
                action.street_total_bb = action.street_total
            if action.street_total_dollars is None and bb:
                action.street_total_dollars = round(action.street_total * bb, 2)

    # ── round accumulation ────────────────────────────────────────────────────

    def _new_round(self, street: Street) -> ActionRound:
        self._round_counter += 1
        return ActionRound(street=street, round_number=self._round_counter)

    def _close_round(self):
        if self._current_round and self._hand_state:
            r = self._current_round
            if r.actions_before_hero or r.hero_action or r.actions_after_hero:
                self._hand_state.action_rounds.append(r)
        self._current_round = None

    def _route_action(self, action: PlayerAction):
        """Append action to the correct slot of the current round."""
        r = self._current_round
        if r is None:
            return
        if action.seat == 0:
            r.hero_action = action
        elif r.hero_action is None:
            r.actions_before_hero.append(action)
        else:
            r.actions_after_hero.append(action)

    # ── hand completion ───────────────────────────────────────────────────────

    def _end_hand(self):
        self._close_round()
        hs = self._hand_state
        if hs:
            import time as _t
            hs.completed_at = int(_t.time())
            self._on_hand_complete(hs)
        self._state = _State.IDLE
        self._reset_state()

    def _reset_state(self):
        self._hand_state          = None
        self._hero_present_frames = 0
        self._hero_absent_frames  = 0
        self._btn_was_visible     = False
        self._hero_stack_before   = None
        self._current_round       = None
        self._round_counter       = 0
        self._actions.reset()
        self._streets.reset()
        self._btn_cache.reset()


# ── helpers ───────────────────────────────────────────────────────────────────

def _action_buttons_visible(img: Image.Image, cfg: TrackerConfig) -> bool:
    """Saturation gate on the action button region (~1ms)."""
    if not cfg.regions.action:
        return False
    crop = crop_region(img, cfg.regions.action)
    arr  = np.array(crop.convert("HSV") if hasattr(Image, "HSV")
                    else crop.convert("RGB"), dtype=np.float32)
    # Use the green channel peak as saturation proxy —
    # action buttons (Fold=red, Call=green, Raise=blue) have high colour saturation.
    rgb = np.array(crop.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    mx  = np.maximum(np.maximum(r, g), b)
    mn  = np.minimum(np.minimum(r, g), b)
    safe_mx = np.where(mx > 0, mx, 1.0)
    sat = np.where(mx > 0, (mx - mn) / safe_mx, 0.0)
    return float((sat > 0.4).mean()) > GameTracker.BTN_SAT_THRESHOLD
