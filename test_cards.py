#!/usr/bin/env python3
"""Card detection test — hero cards + community cards.

Runs both detectors on every image in the target directory and prints
a unified results table with separate pass/fail columns for hero and board.

Usage:
    python3 test_cards.py                            # default: captures/
    python3 test_cards.py --images captures/
    python3 test_cards.py --images tests_phase3/ --labels label_cards.json
    python3 test_cards.py --images captures/ --limit 20
    python3 test_cards.py --config regions.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from core.config import load_config
from detection.card_detector import detect_hero_cards
from detection.board_detector import detect_board_cards
from tracking.logger import PokerLogger

_c = PokerLogger.colour


def _norm(s: str) -> str:
    return s[:-1].upper() + s[-1].lower() if len(s) >= 2 else s


# ── columns ───────────────────────────────────────────────────────────────────
_W_FILE  = 26
_W_HERO  = 9     # detected hero
_W_CONF  = 5     # hero confidence
_W_HR    = 8     # Hero? result
_W_BOARD = 22    # detected board
_W_BR    = 20    # Board? result (wide — shows expected when wrong)
_W_MS    = 4

_HEADER = (
    f"  {'Image':<{_W_FILE}}  {'Hero':<{_W_HERO}}  {'Conf':>{_W_CONF}}  "
    f"{'Hero?':<{_W_HR}}  {'Board':<{_W_BOARD}}  {'Board?':<{_W_BR}}  {'ms':>{_W_MS}}"
)
_SEP = "  " + "─" * (len(_HEADER) - 2)


def _print_row(
    name:      str,
    hero_str:  str,
    conf_str:  str,
    hero_res:  str,  # OK / MISS / !Ah Ks / —
    board_str: str,
    board_res: str,  # OK / !Ac 3c 6c / —
    ms:        float,
    hero_col:  str = "",
    board_col: str = "",
):
    h_part = _c(f"{hero_res:<{_W_HR}}", hero_col)  if hero_col  else f"{hero_res:<{_W_HR}}"
    b_part = _c(f"{board_res:<{_W_BR}}", board_col) if board_col else f"{board_res:<{_W_BR}}"
    print(
        f"  {name:<{_W_FILE}}  {hero_str:<{_W_HERO}}  {conf_str:>{_W_CONF}}  "
        f"{h_part}  {board_str:<{_W_BOARD}}  {b_part}  {ms:>{_W_MS}.0f}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hero + community card detection test")
    parser.add_argument("--images",  default="captures/",        help="Directory with PNG images (default: captures/)")
    parser.add_argument("--labels",  default="label_cards.json", help="Label JSON from label_cards.py")
    parser.add_argument("--config",  default="regions.json",     help="Regions config path")
    parser.add_argument("--limit",   type=int, default=0,        help="Stop after N images (0 = all)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        print(f"Directory not found: {img_dir}")
        sys.exit(1)

    images = sorted(
        img_dir.glob("*.png"),
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
    )
    if args.limit:
        images = images[:args.limit]
    if not images:
        print(f"No PNG images found in {img_dir}")
        sys.exit(0)

    # ── labels ────────────────────────────────────────────────────────────────
    labels: dict = {}
    label_file = Path(args.labels)
    if label_file.exists():
        labels = json.loads(label_file.read_text())
        print(f"  Loaded {len(labels)} labels from {label_file}")

    # ── header ────────────────────────────────────────────────────────────────
    print(f"\n  {_c('Card Detection Test', 'bold')}  —  {img_dir}  ({len(images)} images)\n")
    print(_HEADER)
    print(_SEP)

    # ── counters ──────────────────────────────────────────────────────────────
    total = hero_det = board_det = 0
    h_correct = h_wrong = h_missed = 0
    b_correct = b_wrong = b_missed = 0
    total_ms = 0.0
    errors: list = []

    for img_path in images:
        total += 1
        label = labels.get(img_path.name)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            errors.append((img_path.name, str(e)))
            _print_row(img_path.name, "ERROR", "", "", "", "", 0, "red")
            continue

        t0 = time.monotonic()
        try:
            hero_res = detect_hero_cards(img, cfg)
        except Exception as e:
            hero_res = None
            errors.append((img_path.name, f"hero: {e}"))

        try:
            board_raw = detect_board_cards(img, cfg)
        except Exception as e:
            board_raw = []
            errors.append((img_path.name, f"board: {e}"))

        ms = (time.monotonic() - t0) * 1000
        total_ms += ms

        # ── format detected ───────────────────────────────────────────────────
        if hero_res:
            hero_det += 1
            c1 = _norm(hero_res.cards[0].to_str())
            c2 = _norm(hero_res.cards[1].to_str())
            hero_str = f"{c1} {c2}"
            conf_str = f"{hero_res.confidence:.2f}"
        else:
            hero_str = "—"
            conf_str = ""
            c1 = c2 = ""

        board_present = [c for c in board_raw if c is not None]
        if board_present:
            board_det += 1
            board_str = " ".join(board_present)
        else:
            board_str = "—"

        # ── score against labels ──────────────────────────────────────────────
        hero_res_str  = "—"
        board_res_str = "—"
        hero_col      = ""
        board_col     = ""

        if label is not None:
            exp_hero  = label.get("hero")
            exp_board = label.get("board")

            # Hero
            if exp_hero is None:
                if hero_res:
                    hero_res_str = "FP"
                    hero_col     = "red"
                    h_wrong += 1
                else:
                    hero_res_str = "OK"
                    hero_col     = "green"
                    h_correct += 1
            else:
                exp1, exp2 = _norm(exp_hero[0]), _norm(exp_hero[1])
                if hero_res is None:
                    hero_res_str = "MISS"
                    hero_col     = "yellow"
                    h_missed += 1
                elif c1 == exp1 and c2 == exp2:
                    hero_res_str = "OK"
                    hero_col     = "green"
                    h_correct += 1
                else:
                    hero_res_str = f"!{exp1} {exp2}"
                    hero_col     = "red"
                    h_wrong += 1

            # Board
            if exp_board is None:
                if board_present:
                    board_res_str = "FP"
                    board_col     = "red"
                    b_wrong += 1
                else:
                    board_res_str = "OK"
                    board_col     = "green"
                    b_correct += 1
            else:
                exp_b = [_norm(c) for c in exp_board]
                det_b = [_norm(c) for c in board_present]
                if not board_present:
                    board_res_str = "MISS"
                    board_col     = "yellow"
                    b_missed += 1
                elif det_b == exp_b:
                    board_res_str = "OK"
                    board_col     = "green"
                    b_correct += 1
                else:
                    board_res_str = "!" + " ".join(exp_b)
                    board_col     = "red"
                    b_wrong += 1

        _print_row(
            img_path.name,
            hero_str, conf_str, hero_res_str,
            board_str, board_res_str,
            ms,
            hero_col, board_col,
        )

    # ── summary ───────────────────────────────────────────────────────────────
    print(_SEP)
    avg_ms = total_ms / max(total, 1)

    print(f"\n  {'Images tested':<24}: {total}")
    print(f"  {'Hero detected':<24}: {hero_det}  ({hero_det/max(total,1)*100:.0f}%)")
    print(f"  {'Board detected':<24}: {board_det}  ({board_det/max(total,1)*100:.0f}%)")
    print(f"  {'Avg latency':<24}: {avg_ms:.0f}ms")

    if labels:
        h_lab = h_correct + h_wrong + h_missed
        b_lab = b_correct + b_wrong + b_missed
        print()
        if h_lab:
            h_acc = h_correct / h_lab * 100
            print(f"  {'Hero accuracy':<24}: {_c(f'{h_correct}/{h_lab}', 'green')}  ({h_acc:.0f}%)"
                  + (f"  wrong={_c(str(h_wrong), 'red')}" if h_wrong else "")
                  + (f"  missed={_c(str(h_missed), 'yellow')}" if h_missed else ""))
        if b_lab:
            b_acc = b_correct / b_lab * 100
            print(f"  {'Board accuracy':<24}: {_c(f'{b_correct}/{b_lab}', 'green')}  ({b_acc:.0f}%)"
                  + (f"  wrong={_c(str(b_wrong), 'red')}" if b_wrong else "")
                  + (f"  missed={_c(str(b_missed), 'yellow')}" if b_missed else ""))

    if errors:
        print(f"\n  {_c(f'Errors ({len(errors)})', 'red')}")
        for name, msg in errors:
            print(f"    {name}: {msg}")

    print()


if __name__ == "__main__":
    main()
