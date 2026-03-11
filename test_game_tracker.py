#!/usr/bin/env python3
"""Test the full GameTracker pipeline in replay mode.

Processes sorted PNGs from a test directory, prints all events as they fire,
and shows a hand summary table at the end.

Usage:
    python3 test_game_tracker.py
    python3 test_game_tracker.py --tests path/to/tests/
    python3 test_game_tracker.py --tests path/to/tests/ --limit 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

from core.config import load_config
from tracking.game_tracker import GameTracker
from core.hand_state import HandState, PlayerAction, Street
from core.models import HeroCardsResult
from core.table_state import TableState

# ── event handlers ────────────────────────────────────────────────────────────

_hands: list[HandState] = []
_event_count = 0


def _on_hand_start(result: HeroCardsResult, table: TableState):
    global _event_count
    _event_count += 1
    cards = " ".join(c.to_str() for c in result.cards)
    dealer = (
        "hero" if table.dealer_seat == 0
        else f"seat {table.dealer_seat}"
        if table.dealer_seat is not None else "?"
    )
    print(f"\n{'━'*64}")
    print(f"  HAND_START  cards={cards}  dealer={dealer}  "
          f"pos={table.hero_position or '?'}  hand={table.hand_id}")


def _on_street_change(street: Street):
    global _event_count
    _event_count += 1
    print(f"  ── {street} {'─'*50}")


def _on_player_action(action: PlayerAction):
    global _event_count
    _event_count += 1
    seat   = "hero" if action.seat == 0 else f"seat {action.seat}"
    name   = f"({action.username})" if action.username else ""
    amount = f"  {action.amount:.0f}" if action.amount is not None else ""
    bb_str = f"  ({action.amount_bb:.1f}BB)" if action.amount_bb is not None else ""
    print(f"  PLAYER_ACTION  {seat:<8} {name:<22} {action.action:<8}{amount}{bb_str}")


def _on_your_turn(hand: HandState):
    global _event_count
    _event_count += 1
    before = hand.all_actions[-5:] if hand.all_actions else []
    summary = ", ".join(
        f"seat{a.seat} {a.action}" + (f" {a.amount:.0f}" if a.amount else "")
        for a in before
        if a.seat != 0
    ) or "none"
    print(f"  YOUR_TURN   context=[{summary}]")


def _on_hand_complete(hand: HandState):
    global _event_count
    _event_count += 1
    _hands.append(hand)
    rounds  = len(hand.action_rounds)
    actions = len(hand.all_actions)
    folded  = "  (hero folded)" if hand.hero_folded else ""
    board   = " ".join(hand.community_cards) if hand.community_cards else "—"
    bb_str  = f"  BB={hand.bb_amount}" if hand.bb_amount else ""
    print(f"  HAND_COMPLETE  {rounds} rounds  {actions} actions  board=[{board}]{bb_str}{folded}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GameTracker replay test")
    parser.add_argument("--tests",  default="tests/")
    parser.add_argument("--config", default="regions.json")
    parser.add_argument("--room",   default="PS")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Stop after N images (0 = all)")
    parser.add_argument("--json",   metavar="FILE",
                        help="Also write full hand history to FILE as JSON")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    tracker = GameTracker(
        cfg,
        room              = args.room,
        on_hand_start     = _on_hand_start,
        on_street_change  = _on_street_change,
        on_player_action  = _on_player_action,
        on_your_turn      = _on_your_turn,
        on_hand_complete  = _on_hand_complete,
    )

    # Replay
    from PIL import Image
    p      = Path(args.tests)
    images = sorted(
        p.glob("*.png"),
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
    )
    if args.limit:
        images = images[:args.limit]

    print(f"Replaying {len(images)} images from {args.tests}…")
    t0 = time.monotonic()

    for img_path in images:
        try:
            tracker.process_frame(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  ERROR {img_path.name}: {e}", file=sys.stderr)

    elapsed = time.monotonic() - t0

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'━'*64}")
    print(f"  SUMMARY")
    print(f"{'━'*64}")
    print(f"  Images processed : {len(images)}  ({elapsed*1000/max(len(images),1):.0f}ms avg)")
    print(f"  Hands detected   : {len(_hands)}")
    print(f"  Total events     : {_event_count}")

    if _hands:
        print(f"\n  {'Hand':<12} {'Cards':<8} {'BB':>6}  {'Pos':<5} {'Board':<16} "
              f"{'Acts':<6} {'Folded'}")
        print(f"  {'─'*10}  {'─'*6}  {'─'*5}  {'─'*4}  {'─'*14}  {'─'*4}  {'─'*6}")
        for h in _hands:
            cards  = " ".join(c.to_str() for c in h.hero_cards)
            pos    = h.table_state.hero_position if h.table_state else "?"
            board  = " ".join(h.community_cards) if h.community_cards else "—"
            bb_s   = f"{h.bb_amount}" if h.bb_amount else "?"
            acts   = len(h.all_actions)
            folded = "yes" if h.hero_folded else "no"
            print(f"  {h.hand_id:<12} {cards:<8} {bb_s:>6}  {pos or '?':<5} "
                  f"{board:<16} {acts:<6} {folded}")

    # ── optional JSON dump ────────────────────────────────────────────────────
    if args.json:
        out = [h.to_dict() for h in _hands]
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n  Hand history written → {args.json}")

    print()


if __name__ == "__main__":
    main()
