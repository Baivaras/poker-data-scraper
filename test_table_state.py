#!/usr/bin/env python3
"""Test table state detection on replay images.

Modes:
  --mode print   (default) Run on all images, print a formatted table per image.
                 Use this first to verify output looks correct.
  --mode label   After manual verification, write confirmed results to
                 label_table_state.json for future regression testing.
  --mode test    Compare live output against label_table_state.json and
                 report accuracy. (Use after labeling.)

Usage:
    python3 test_table_state.py
    python3 test_table_state.py --mode print --limit 30
    python3 test_table_state.py --mode label
    python3 test_table_state.py --mode test
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from core.config import load_config
from tracking.table_scanner import TableScanner
from core.table_state import TableState

LABEL_FILE = "label_table_state.json"
TESTS_DIR  = "tests"


# ── helpers ───────────────────────────────────────────────────────────────────

def _sorted_images(tests_dir: str) -> list[Path]:
    p = Path(tests_dir)
    return sorted(
        p.glob("*.png"),
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
    )


def _print_state(state: TableState, img_name: str, elapsed_ms: float):
    """Pretty-print one TableState."""
    dealer_label = (
        "hero" if state.dealer_seat == 0
        else f"seat {state.dealer_seat}"
        if state.dealer_seat is not None else "unknown"
    )
    print(f"\n{'─'*62}")
    print(f"  {img_name:<20}  dealer={dealer_label:<10}  {elapsed_ms:.0f}ms")
    print(f"  hero: {state.hero_username or '?':20}  pos={state.hero_position or '?'}")
    print(f"  {'Seat':<6}{'Username':<22}{'Stack':>10}  {'Status':<12}{'Pos'}")
    print(f"  {'─'*4}  {'─'*20}  {'─'*8}  {'─'*10}  {'─'*5}")
    for p in state.players:
        seat_lbl  = "hero" if p.is_hero else str(p.seat)
        stack_str = f"{p.stack:>8.0f}" if p.stack is not None else "       ?"
        print(
            f"  {seat_lbl:<6}{(p.username or ''):22}{stack_str}  "
            f"{p.status:<12}{p.position or ''}"
        )


def _state_to_label(state: TableState, img_name: str) -> dict:
    return {
        "image":        img_name,
        "dealer_seat":  state.dealer_seat,
        "hero_position": state.hero_position,
        "players": [
            {
                "seat":     p.seat,
                "username": p.username,
                "stack":    p.stack,
                "status":   p.status,
                "position": p.position,
            }
            for p in state.players
        ],
    }


# ── modes ─────────────────────────────────────────────────────────────────────

def mode_print(scanner: TableScanner, tests_dir: str, limit: int):
    images = _sorted_images(tests_dir)
    if limit:
        images = images[:limit]
    print(f"Scanning {len(images)} images from {tests_dir}…")
    t0 = time.monotonic()
    for img_path in images:
        try:
            img   = Image.open(img_path).convert("RGB")
            t1    = time.monotonic()
            state = scanner.scan(img, hand_id=img_path.stem)
            ms    = (time.monotonic() - t1) * 1000
            _print_state(state, img_path.name, ms)
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}", file=sys.stderr)
    total = (time.monotonic() - t0) * 1000
    print(f"\n{'─'*62}")
    print(f"  Done. {len(images)} images in {total:.0f}ms  ({total/max(len(images),1):.1f}ms avg)")


def mode_label(scanner: TableScanner, tests_dir: str):
    images  = _sorted_images(tests_dir)
    labels  = []
    print(f"Scanning {len(images)} images to build labels…")
    for img_path in images:
        try:
            img   = Image.open(img_path).convert("RGB")
            state = scanner.scan(img, hand_id=img_path.stem)
            labels.append(_state_to_label(state, img_path.name))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}", file=sys.stderr)
    Path(LABEL_FILE).write_text(json.dumps(labels, indent=2))
    print(f"Wrote {len(labels)} labels → {LABEL_FILE}")
    print("Review the file, remove/correct any wrong entries, then run --mode test.")


def mode_test(scanner: TableScanner, tests_dir: str):
    if not Path(LABEL_FILE).exists():
        print(f"No label file found at {LABEL_FILE}. Run --mode label first.")
        sys.exit(1)

    labels    = {e["image"]: e for e in json.loads(Path(LABEL_FILE).read_text())}
    images    = [p for p in _sorted_images(tests_dir) if p.name in labels]
    total     = len(images)
    wrong     = []

    for img_path in images:
        label = labels[img_path.name]
        try:
            img   = Image.open(img_path).convert("RGB")
            state = scanner.scan(img, hand_id=img_path.stem)
        except Exception as e:
            wrong.append({"image": img_path.name, "field": "ERROR", "expected": "", "got": str(e)})
            continue

        # Compare dealer_seat
        if state.dealer_seat != label["dealer_seat"]:
            wrong.append({
                "image":    img_path.name,
                "field":    "dealer_seat",
                "expected": label["dealer_seat"],
                "got":      state.dealer_seat,
            })

        # Compare hero_position
        if state.hero_position != label["hero_position"]:
            wrong.append({
                "image":    img_path.name,
                "field":    "hero_position",
                "expected": label["hero_position"],
                "got":      state.hero_position,
            })

        # Compare player statuses
        label_players = {p["seat"]: p for p in label["players"]}
        for p in state.players:
            lp = label_players.get(p.seat)
            if not lp:
                continue
            if p.status != lp["status"]:
                wrong.append({
                    "image":    img_path.name,
                    "field":    f"seat{p.seat}.status",
                    "expected": lp["status"],
                    "got":      p.status,
                })

    # ── summary ───────────────────────────────────────────────────────────────
    correct = total - len({w["image"] for w in wrong})
    print("=" * 64)
    print("  TABLE STATE TEST RESULTS")
    print("=" * 64)
    print(f"  Images tested : {total}")
    print(f"  Fully correct : {correct}  ({correct/max(total,1)*100:.1f}%)")
    print(f"  With errors   : {len({w['image'] for w in wrong})}")
    print("=" * 64)

    if not wrong:
        print("\n  All images matched labels perfectly.")
        return

    # ── error table ───────────────────────────────────────────────────────────
    print(f"\n  MISMATCHES ({len(wrong)} total)\n")
    cw_img   = max(len("Image"),   max(len(w["image"]) for w in wrong))
    cw_field = max(len("Field"),   max(len(w["field"]) for w in wrong))
    cw_exp   = max(len("Expected"),max(len(str(w["expected"])) for w in wrong))
    cw_got   = max(len("Got"),     max(len(str(w["got"])) for w in wrong))

    sep = f"  +-{'-'*cw_img}-+-{'-'*cw_field}-+-{'-'*cw_exp}-+-{'-'*cw_got}-+"
    print(sep)
    print(f"  | {'Image':<{cw_img}} | {'Field':<{cw_field}} | {'Expected':<{cw_exp}} | {'Got':<{cw_got}} |")
    print(sep)
    for w in wrong:
        print(
            f"  | {w['image']:<{cw_img}} | {w['field']:<{cw_field}} |"
            f" {str(w['expected']):<{cw_exp}} | {str(w['got']):<{cw_got}} |"
        )
    print(sep)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Table state detection test")
    parser.add_argument("--mode",    default="print",
                        choices=["print", "label", "test"])
    parser.add_argument("--tests",   default=TESTS_DIR)
    parser.add_argument("--config",  default="regions.json")
    parser.add_argument("--room",    default="PS")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Limit number of images (print mode only)")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    scanner = TableScanner(cfg, room=args.room)

    if args.mode == "print":
        mode_print(scanner, args.tests, args.limit)
    elif args.mode == "label":
        mode_label(scanner, args.tests)
    elif args.mode == "test":
        mode_test(scanner, args.tests)


if __name__ == "__main__":
    main()
