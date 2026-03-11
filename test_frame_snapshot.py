#!/usr/bin/env python3
"""Per-frame snapshot test — generates a JSON with all tracked values per image.

Loops through every PNG in a test directory, runs the full detection stack
(hero cards, table state, board cards), and writes one JSON entry per image.

Useful for:
  - Verifying OCR accuracy across a session
  - Debugging what the tracker sees on each frame
  - Building ground truth for regression tests

Usage:
    python3 test_frame_snapshot.py
    python3 test_frame_snapshot.py --tests tests_phase3/ --out snapshot.json
    python3 test_frame_snapshot.py --tests tests_phase3/ --limit 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from core.config import load_config
from detection.card_detector import detect_hero_cards
from detection.board_detector import detect_board_cards
from detection.street_detector import count_board_cards, card_count_to_street
from tracking.table_scanner import TableScanner


# ── per-frame snapshot ────────────────────────────────────────────────────────

def snapshot_frame(img: Image.Image, scanner: TableScanner, cfg) -> dict:
    """Run all detectors on one frame and return a flat dict."""
    result = {}

    # ── hero cards ────────────────────────────────────────────────────────────
    hero = detect_hero_cards(img, cfg)
    if hero:
        result["hero_cards"]       = [c.to_str() for c in hero.cards]
        result["hero_cards_conf"]  = hero.confidence
    else:
        result["hero_cards"]       = None
        result["hero_cards_conf"]  = None

    # ── table state (dealer, stacks, usernames, positions) ───────────────────
    try:
        state = scanner.scan(img, hand_id="snap")
        result["dealer_seat"]   = state.dealer_seat
        result["hero_username"] = state.hero_username
        result["hero_position"] = state.hero_position
        result["hero_stack"]    = state.hero_stack if hasattr(state, "hero_stack") else _hero_stack(state)
        result["players"] = [
            {
                "seat":     p.seat,
                "username": p.username,
                "stack":    p.stack,
                "status":   p.status,
                "position": p.position,
                "is_hero":  p.is_hero,
            }
            for p in state.players
        ]
    except Exception as e:
        result["table_error"] = str(e)
        result["dealer_seat"]   = None
        result["hero_username"] = None
        result["hero_position"] = None
        result["hero_stack"]    = None
        result["players"]       = []

    # ── board ─────────────────────────────────────────────────────────────────
    board_count  = count_board_cards(img, cfg)
    board_street = card_count_to_street(board_count)
    board_cards  = detect_board_cards(img, cfg)

    result["board_count"]   = board_count
    result["board_street"]  = board_street
    result["board_cards"]   = [c if c else None for c in board_cards]

    return result


def _hero_stack(state) -> float | None:
    """Extract hero stack from players list."""
    for p in state.players:
        if p.is_hero:
            return p.stack
    return None


# ── printing ──────────────────────────────────────────────────────────────────

def _print_frame(image_name: str, snap: dict, elapsed_ms: float):
    hero_cards = " ".join(snap["hero_cards"]) if snap["hero_cards"] else "—"
    conf       = f"({snap['hero_cards_conf']:.2f})" if snap["hero_cards_conf"] else ""
    dealer     = (f"seat {snap['dealer_seat']}" if snap["dealer_seat"] is not None
                  else "?")
    board      = " ".join(c or "?" for c in snap["board_cards"]) if snap["board_cards"] else "—"

    print(f"\n  {'─'*62}")
    print(f"  {image_name:<20}  {elapsed_ms:.0f}ms  "
          f"street={snap['board_street']}  dealer={dealer}")
    print(f"  hero: cards={hero_cards:<8}{conf}  "
          f"pos={snap['hero_position'] or '?'}  "
          f"stack={snap['hero_stack'] or '?'}  "
          f"name={snap['hero_username'] or '?'}")
    if board != "—":
        print(f"  board: {board}")
    if snap["players"]:
        print(f"  {'Seat':<6} {'Username':<22} {'Stack':>8}  {'Status':<12} {'Pos'}")
        for p in snap["players"]:
            seat_lbl  = "hero" if p["is_hero"] else str(p["seat"])
            stack_str = f"{p['stack']:>8.0f}" if p["stack"] is not None else "       ?"
            print(f"  {seat_lbl:<6} {(p['username'] or ''):22} {stack_str}  "
                  f"{p['status']:<12} {p['position'] or ''}")
    if "table_error" in snap:
        print(f"  TABLE ERROR: {snap['table_error']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-frame snapshot test")
    parser.add_argument("--tests",  default="tests_phase3/",
                        help="Directory with PNG images (default: tests_phase3/)")
    parser.add_argument("--config", default="regions.json")
    parser.add_argument("--room",   default="PS")
    parser.add_argument("--out",    default="snapshot.json",
                        help="Output JSON file (default: snapshot.json)")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Stop after N images (0 = all)")
    parser.add_argument("--quiet",  action="store_true",
                        help="Skip per-frame console output, only write JSON")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    scanner = TableScanner(cfg, room=args.room)

    test_dir = Path(args.tests)
    if not test_dir.is_dir():
        print(f"Directory not found: {test_dir}")
        sys.exit(1)

    images = sorted(
        test_dir.glob("*.png"),
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
    )
    if args.limit:
        images = images[:args.limit]

    print(f"Scanning {len(images)} images from {test_dir}…")
    t0      = time.monotonic()
    results = []

    for img_path in images:
        try:
            img  = Image.open(img_path).convert("RGB")
            t1   = time.monotonic()
            snap = snapshot_frame(img, scanner, cfg)
            ms   = (time.monotonic() - t1) * 1000
            snap["image"]      = img_path.name
            snap["elapsed_ms"] = round(ms, 1)
            results.append(snap)
            if not args.quiet:
                _print_frame(img_path.name, snap, ms)
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}", file=sys.stderr)
            results.append({"image": img_path.name, "error": str(e)})

    total_ms = (time.monotonic() - t0) * 1000

    # ── summary ───────────────────────────────────────────────────────────────
    detected_cards  = sum(1 for r in results if r.get("hero_cards"))
    detected_dealer = sum(1 for r in results if r.get("dealer_seat") is not None)
    errors          = sum(1 for r in results if "error" in r or "table_error" in r)

    print(f"\n  {'═'*62}")
    print(f"  SUMMARY")
    print(f"  {'═'*62}")
    print(f"  Images scanned   : {len(results)}")
    print(f"  Avg latency      : {total_ms/max(len(results),1):.0f}ms")
    print(f"  Hero cards found : {detected_cards}/{len(results)}")
    print(f"  Dealer detected  : {detected_dealer}/{len(results)}")
    print(f"  Errors           : {errors}")

    # ── write JSON ────────────────────────────────────────────────────────────
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n  Snapshot written → {args.out}")
    print(f"  Open it to inspect per-frame values for any image.\n")


if __name__ == "__main__":
    main()
