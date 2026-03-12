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
import time
from pathlib import Path

from core.config import load_config
from detection.capture import get_pokerstars_window, screenshot
from tracking.game_tracker import GameTracker
from tracking.logger import PokerLogger


# ── screenshot capture ────────────────────────────────────────────────────────

class ScreenshotCapture:
    """Listens for SPACE key via pynput (no terminal mode changes) and saves a screenshot."""

    def __init__(self, output_dir: Path, logger: PokerLogger):
        self._dir    = output_dir
        self._logger = logger
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
            self._logger.log("CAPTURE", f"Key listener unavailable: {e}  (SPACE disabled)", "dim")

    def _capture(self):
        win = get_pokerstars_window()
        if win is None:
            self._logger.log("CAPTURE", "PokerStars window not found", "red")
            return
        img = screenshot(win)
        if img is None:
            self._logger.log("CAPTURE", "Screenshot failed", "red")
            return
        name = f"cap_{time.strftime('%Y%m%d_%H%M%S')}.png"
        img.save(self._dir / name)
        self._logger.log("CAPTURE", f"Saved → {self._dir}/{name}", "cyan")


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

    logger = PokerLogger(verbose=args.verbose)

    if args.debug or args.verbose:
        log_path = logger.setup_log_file()
        print(f"  Logging to  : {log_path}")

    cfg = load_config(args.config)

    capture = ScreenshotCapture(Path(args.screenshots), logger)
    capture.start()

    tracker = GameTracker(
        cfg,
        room             = args.room,
        on_hand_start    = logger.on_hand_start,
        on_street_change = logger.on_street_change,
        on_player_action = logger.on_player_action,
        on_your_turn     = logger.on_your_turn,
        on_hand_complete = logger.on_hand_complete,
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
