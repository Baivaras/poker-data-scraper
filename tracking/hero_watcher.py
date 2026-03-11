#!/usr/bin/env python3
"""HeroWatcher: polling loop that detects new hole cards and emits them.

Usage (live):
    cfg = load_config()
    watcher = HeroWatcher(cfg, on_new_hand=lambda r: print(r.to_dict()))
    watcher.start()

Usage (test replay):
    watcher.replay_dir("../pkr-ui-reader-main/tests/")

CLI test:
    python3 hero_watcher.py --test <dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from detection.capture import get_pokerstars_window, screenshot
from detection.card_detector import detect_hero_cards
from core.config import TrackerConfig, crop_region, load_config
from core.models import HeroCardsResult

import numpy as np


def _variance(img: Image.Image) -> float:
    arr = np.array(img.convert("L"), dtype=float)
    return float(arr.std())


class HeroWatcher:
    """Poll the PokerStars window and emit HeroCardsResult on each new hand.

    Debounce: requires `cfg.debounce_frames` consecutive frames with cards
    present before running the NN classifier.

    Dedup: only emits if the card pair changed since last emission (prevents
    re-emitting the same hand every frame).
    """

    def __init__(self, cfg: TrackerConfig, on_new_hand: Callable[[HeroCardsResult], None]):
        self._cfg          = cfg
        self._on_new_hand  = on_new_hand
        self._running      = False

        # Debounce state
        self._consecutive_present = 0  # frames in a row with cards visible
        self._consecutive_absent  = 0

        # Dedup: last emitted card pair string e.g. "AhKs"
        self._last_emitted: Optional[str] = None

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self):
        """Live mode: poll the PS window at cfg.poll_ms intervals.

        Blocks until KeyboardInterrupt.
        """
        print(f"[HeroWatcher] Polling every {self._cfg.poll_ms}ms — waiting for PokerStars window...",
              file=sys.stderr)
        interval = self._cfg.poll_ms / 1000.0
        self._running = True
        try:
            while self._running:
                t0  = time.monotonic()
                win = get_pokerstars_window()
                if win is None:
                    print("[HeroWatcher] PokerStars window not found — retrying...", file=sys.stderr)
                else:
                    img = screenshot(win)
                    if img:
                        self.process_frame(img)
                elapsed = time.monotonic() - t0
                sleep_s = max(0.0, interval - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[HeroWatcher] Stopped.", file=sys.stderr)
        finally:
            self._running = False

    def stop(self):
        self._running = False

    def process_frame(self, img: Image.Image) -> None:
        """Process a single frame. Updates internal state and emits if cards detected."""
        cfg = self._cfg

        # Fast variance gate on hero region
        hero_crop = crop_region(img, cfg.regions.hero)
        v = _variance(hero_crop)
        cards_visible = v >= cfg.card_variance_threshold

        if cards_visible:
            self._consecutive_present += 1
            self._consecutive_absent   = 0
        else:
            self._consecutive_absent  += 1
            self._consecutive_present  = 0
            # Reset dedup when cards disappear (hand ended)
            if self._consecutive_absent == 1:
                self._last_emitted = None
            return

        # Debounce: need N consecutive frames before reading
        if self._consecutive_present < cfg.debounce_frames:
            return

        # Run NN
        result = detect_hero_cards(img, cfg)
        if result is None:
            return

        # Dedup: skip if same hand
        key = "".join(c.to_str() for c in result.cards)
        if key == self._last_emitted:
            return

        self._last_emitted = key
        self._on_new_hand(result)

    def replay_dir(self, path: str) -> None:
        """Test mode: process all PNGs in path (sorted), call on_new_hand per detection.

        Resets watcher state before replay.
        """
        self._reset_state()
        p = Path(path)
        images = sorted(
            p.glob("*.png"),
            key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
        )
        if not images:
            print(f"[HeroWatcher] No PNG files found in {path}", file=sys.stderr)
            return
        print(f"[HeroWatcher] Replaying {len(images)} images from {path}", file=sys.stderr)
        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                self.process_frame(img)
            except Exception as e:
                print(f"[HeroWatcher] Error processing {img_path.name}: {e}", file=sys.stderr)

    # ── internals ─────────────────────────────────────────────────────────────

    def _reset_state(self):
        self._consecutive_present = 0
        self._consecutive_absent  = 0
        self._last_emitted        = None


# ── CLI entry point ────────────────────────────────────────────────────────────

def _on_hand_cli(result: HeroCardsResult):
    print(json.dumps(result.to_dict(), indent=2))


def main():
    parser = argparse.ArgumentParser(description="Hero Card Watcher — Phase 1")
    parser.add_argument("--test", metavar="DIR",
                        help="Replay mode: process PNGs in DIR and print detections")
    parser.add_argument("--config", default="regions.json",
                        help="Path to regions.json (default: regions.json)")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    watcher = HeroWatcher(cfg, on_new_hand=_on_hand_cli)

    if args.test:
        watcher.replay_dir(args.test)
    else:
        watcher.start()


if __name__ == "__main__":
    main()
