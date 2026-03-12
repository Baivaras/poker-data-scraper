#!/usr/bin/env python3
"""Interactive card labeller — hero + community cards.

Opens each image, runs detection, and lets you confirm or correct the result.
Labels are saved to a JSON file after every image so progress is never lost.
Resume anytime — already-labelled images are skipped by default.

Usage:
    python3 label_cards.py                        # captures/, saves label_cards.json
    python3 label_cards.py --images captures/
    python3 label_cards.py --labels my_labels.json
    python3 label_cards.py --redo                 # re-label already labelled images too

Controls (at each prompt):
    Enter          Accept detected value
    Ah Ks          Override with these cards (space-separated)
    -              Mark as not present / empty
    s              Skip this image (no label saved)
    b              Go back to previous image
    q              Quit and save
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from core.config import load_config
from detection.card_detector import detect_hero_cards
from detection.board_detector import detect_board_cards


# ── ANSI ──────────────────────────────────────────────────────────────────────
_R      = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_RED    = "\033[91m"

def _c(t, code): return f"{code}{t}{_R}"


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s[:-1].upper() + s[-1].lower() if len(s) >= 2 else s.upper()


def _parse_cards(raw: str) -> Optional[list]:
    """Parse user input into a list of normalised card strings, or None."""
    raw = raw.strip()
    if raw in ("-", "none", ""):
        return None
    parts = raw.upper().split()
    normed = []
    for p in parts:
        if len(p) < 2:
            return False   # invalid
        normed.append(p[:-1].upper() + p[-1].lower())
    return normed or None


def _open_image(path: Path):
    """Open image in the system viewer (non-blocking)."""
    try:
        subprocess.Popen(["open", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _load_labels(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _save_labels(labels: dict, path: Path):
    path.write_text(json.dumps(labels, indent=2))


# ── prompt ────────────────────────────────────────────────────────────────────

def _prompt(label: str, detected: Optional[list], required: int = 0) -> tuple:
    """Prompt for one field.  Returns (cards_list_or_None, command_str)."""
    det_str = " ".join(detected) if detected else _c("—", _DIM)
    hint    = f"Enter=accept  cards like 'Ah Ks'  -=none  s=skip  b=back  q=quit"

    print(f"  {_c(label, _CYAN)}  detected: {_c(det_str, _YELLOW)}")
    print(f"  {_c(hint, _DIM)}")

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None, "q"

        if raw in ("q", "s", "b"):
            return None, raw

        if raw == "":
            return detected, "ok"   # accept detection

        result = _parse_cards(raw)
        if result is False:
            print(f"  {_c('Invalid — use format like: Ah Ks  or  -', _RED)}")
            continue
        return result, "ok"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive card labeller")
    parser.add_argument("--images",  default="captures/",       help="Directory of PNG images")
    parser.add_argument("--labels",  default="label_cards.json",help="Label output file (JSON)")
    parser.add_argument("--config",  default="regions.json",    help="Regions config")
    parser.add_argument("--redo",    action="store_true",        help="Re-label already labelled images")
    parser.add_argument("--limit",   type=int, default=0,       help="Stop after N images")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    img_dir   = Path(args.images)
    label_path = Path(args.labels)

    if not img_dir.is_dir():
        print(f"Directory not found: {img_dir}")
        sys.exit(1)

    images = sorted(
        img_dir.glob("*.png"),
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
    )
    if args.limit:
        images = images[:args.limit]

    labels = _load_labels(label_path)
    if labels:
        print(f"  Loaded {len(labels)} existing labels from {label_path}")

    # Filter already-labelled unless --redo
    pending = [p for p in images if args.redo or p.name not in labels]
    done    = len(images) - len(pending)
    print(f"  {len(images)} images total  |  {done} already labelled  |  {len(pending)} to label\n")

    if not pending:
        print("  Nothing to label. Use --redo to re-label existing entries.")
        sys.exit(0)

    idx = 0
    while idx < len(pending):
        img_path = pending[idx]
        total_str = f"[{idx + 1 + done}/{len(images)}]"

        # ── run detection ─────────────────────────────────────────────────────
        try:
            img = Image.open(img_path).convert("RGB")
            t0  = time.monotonic()
            hero_res   = detect_hero_cards(img, cfg)
            board_cards = detect_board_cards(img, cfg)
            ms = (time.monotonic() - t0) * 1000
        except Exception as e:
            print(f"  {_c('Error loading image', _RED)}: {e}")
            idx += 1
            continue

        hero_det  = ([_norm(hero_res.cards[0].to_str()),
                      _norm(hero_res.cards[1].to_str())]
                     if hero_res else None)
        board_det = [c for c in board_cards if c is not None] or None
        conf_str  = f"  conf={hero_res.confidence:.2f}" if hero_res else ""

        # ── header ────────────────────────────────────────────────────────────
        print(f"\n  {'─'*60}")
        print(f"  {_c(total_str, _BOLD)}  {_c(img_path.name, _BOLD)}"
              f"{_c(f'  ({ms:.0f}ms{conf_str})', _DIM)}")
        print(f"  {'─'*60}")

        _open_image(img_path)

        # ── hero prompt ───────────────────────────────────────────────────────
        hero_cards, cmd = _prompt("Hero cards  (2 cards)", hero_det)
        if cmd == "q":
            print(f"\n  Saved {len(labels)} labels → {label_path}")
            _save_labels(labels, label_path)
            sys.exit(0)
        if cmd == "s":
            idx += 1
            continue
        if cmd == "b":
            if idx > 0:
                idx -= 1
            continue

        # ── board prompt ──────────────────────────────────────────────────────
        print()
        board_result, cmd = _prompt("Board cards (0-5 cards)", board_det)
        if cmd == "q":
            print(f"\n  Saved {len(labels)} labels → {label_path}")
            _save_labels(labels, label_path)
            sys.exit(0)
        if cmd == "s":
            idx += 1
            continue
        if cmd == "b":
            continue   # re-do hero prompt for same image

        # ── save ──────────────────────────────────────────────────────────────
        labels[img_path.name] = {
            "hero":  hero_cards,
            "board": board_result,
        }
        _save_labels(labels, label_path)

        status = _c("saved", _GREEN)
        h_str  = " ".join(hero_cards)  if hero_cards  else _c("—", _DIM)
        b_str  = " ".join(board_result) if board_result else _c("—", _DIM)
        print(f"\n  {status}  hero={h_str}  board={b_str}")

        idx += 1

    print(f"\n  {'─'*60}")
    print(f"  Done. {len(labels)} labels saved → {label_path}")
    print(f"  {'─'*60}\n")


if __name__ == "__main__":
    main()
