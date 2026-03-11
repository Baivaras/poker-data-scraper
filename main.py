#!/usr/bin/env python3
"""Poker Tracker — live entry point.

Tracks a PokerStars table in real time and logs every event to the terminal.
Press SPACE at any time to save a screenshot of the current table state.

Usage:
    python3 main.py
    python3 main.py --room PS --config regions.json
    python3 main.py --screenshots captures/
"""

import argparse
import re
import sys
import time
from pathlib import Path

from PIL import Image

from core.config import load_config
from core.hand_state import HandState, PlayerAction, Street
from core.models import HeroCardsResult
from core.table_state import TableState
from detection.capture import get_pokerstars_window, screenshot
from tracking.game_tracker import GameTracker


# ── log file tee ──────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


class _TeeWriter:
    """Writes to the original stream AND a log file (ANSI codes stripped for file)."""

    def __init__(self, stream, log_file):
        self._stream   = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(_ANSI_RE.sub("", data))
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_log_file() -> Path:
    log_dir  = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"tracker_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_fh   = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeWriter(sys.stdout, log_fh)
    sys.stderr = _TeeWriter(sys.stderr, log_fh)
    return log_path


# ── logging helpers ───────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(tag: str, msg: str, colour: str = ""):
    RESET  = "\033[0m"
    codes  = {
        "green":  "\033[92m",
        "yellow": "\033[93m",
        "cyan":   "\033[96m",
        "red":    "\033[91m",
        "bold":   "\033[1m",
        "dim":    "\033[2m",
    }
    c = codes.get(colour, "")
    print(f"  {_ts()}  {c}{tag:<16}{RESET}  {msg}", flush=True)


# ── event handlers ────────────────────────────────────────────────────────────

def on_hand_start(result: HeroCardsResult, table: TableState):
    cards  = " ".join(c.to_str() for c in result.cards)
    dealer = ("hero" if table.dealer_seat == 0
              else f"seat {table.dealer_seat}"
              if table.dealer_seat is not None else "?")
    print(f"\n  {'━'*66}", flush=True)
    _log("HAND START", f"cards={cards}  conf={result.confidence:.2f}  "
         f"dealer={dealer}  pos={table.hero_position or '?'}  "
         f"hand={table.hand_id}", "bold")

    # Print seat summary
    active = [p for p in table.players if p.status == "ACTIVE"]
    for p in active:
        seat = "hero" if p.is_hero else f"seat {p.seat}"
        _log("", f"  {seat:<8}  {(p.username or ''):22}  "
             f"stack={p.stack or '?'}  pos={p.position or '?'}", "dim")


def on_street_change(street: Street):
    colours = {"PREFLOP": "dim", "FLOP": "cyan", "TURN": "yellow", "RIVER": "red"}
    _log(f"── {street}", "─" * 48, colours.get(street, ""))


def _fmt_amount(v: float) -> str:
    """Format a number without unnecessary trailing zeros (e.g. 13.5, not 13.50 or 14)."""
    return f"{v:g}"


_verbose = False


def on_player_action(action: PlayerAction):
    seat   = "hero" if action.seat == 0 else f"seat {action.seat}"
    name   = f"({action.username})" if action.username else ""
    amount = f"  {_fmt_amount(action.amount)}BB" if action.amount is not None else ""
    dollar = (f"  (${_fmt_amount(action.amount_dollars)})"
              if action.amount_dollars is not None else "")
    colour = "yellow" if action.seat == 0 else ""

    if _verbose and action.stack_before is not None and action.stack_after is not None:
        stack_info = (f"  [stack {_fmt_amount(action.stack_before)}BB"
                      f" → {_fmt_amount(action.stack_after)}BB"
                      f"  Δ{_fmt_amount(action.stack_before - action.stack_after)}BB]")
    else:
        stack_info = ""

    _log("PLAYER ACTION",
         f"{seat:<8}  {name:<22}  {action.action:<8}{amount}{dollar}{stack_info}",
         colour)


def on_your_turn(hand: HandState):
    recent = hand.all_actions[-5:] if hand.all_actions else []
    context = ", ".join(
        f"seat{a.seat} {a.action}" + (f" {_fmt_amount(a.amount)}BB" if a.amount else "")
        for a in recent if a.seat != 0
    ) or "none"
    _log("YOUR TURN", f"context=[{context}]", "green")


def on_hand_complete(hand: HandState):
    board  = " ".join(hand.community_cards) if hand.community_cards else "—"
    bb_str = f"  BB={hand.bb_amount}" if hand.bb_amount else ""
    folded = "  (hero folded)" if hand.hero_folded else ""
    _log("HAND COMPLETE",
         f"{len(hand.all_actions)} actions  board=[{board}]{bb_str}{folded}",
         "bold")
    print(f"  {'━'*66}\n", flush=True)


# ── screenshot capture ────────────────────────────────────────────────────────

class ScreenshotCapture:
    """Listens for SPACE key via pynput (no terminal mode changes) and saves a screenshot."""

    def __init__(self, output_dir: Path):
        self._dir = output_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.space:
                    self._capture()

            listener = keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
        except Exception as e:
            _log("CAPTURE", f"Key listener unavailable: {e}  (SPACE disabled)", "dim")

    def _capture(self):
        win = get_pokerstars_window()
        if win is None:
            _log("CAPTURE", "PokerStars window not found", "red")
            return
        img = screenshot(win)
        if img is None:
            _log("CAPTURE", "Screenshot failed", "red")
            return
        name = f"cap_{time.strftime('%Y%m%d_%H%M%S')}.png"
        img.save(self._dir / name)
        _log("CAPTURE", f"Saved → {self._dir}/{name}", "cyan")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Poker Tracker — live")
    parser.add_argument("--config",      default="regions.json")
    parser.add_argument("--room",        default="PS")
    parser.add_argument("--screenshots", default="captures/",
                        help="Directory for SPACE-key screenshots (default: captures/)")
    parser.add_argument("--debug",   action="store_true",
                        help="Enable action-detector debug output")
    parser.add_argument("--verbose", action="store_true",
                        help="Show stack before/after and delta next to each action")
    args = parser.parse_args()

    global _verbose
    _verbose = args.verbose

    if args.debug or args.verbose:
        log_path = _setup_log_file()
        print(f"  Logging to  : {log_path}")

    cfg = load_config(args.config)

    # Start SPACE key screenshot listener
    capture = ScreenshotCapture(Path(args.screenshots))
    capture.start()

    tracker = GameTracker(
        cfg,
        room             = args.room,
        on_hand_start    = on_hand_start,
        on_street_change = on_street_change,
        on_player_action = on_player_action,
        on_your_turn     = on_your_turn,
        on_hand_complete = on_hand_complete,
        debug            = args.debug,
    )

    print(f"\n  {'━'*66}")
    print(f"  Poker Tracker  —  live mode")
    print(f"  {'━'*66}")
    print(f"  Config      : {args.config}")
    print(f"  Room        : {args.room}")
    print(f"  Poll        : {cfg.poll_ms}ms")
    print(f"  Screenshots : {args.screenshots}  (press SPACE to capture)")
    print(f"  Stop        : Ctrl+C\n")

    tracker.start()


if __name__ == "__main__":
    main()
