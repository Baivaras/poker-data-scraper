#!/usr/bin/env python3
"""Hero card detection accuracy test.

Runs the full detection pipeline (variance gate → NN) on all test images and
compares results against a label file.  Prints a per-image result table and a
mismatch table at the end.

Usage:
    python3 test_hero_cards.py
    python3 test_hero_cards.py --labels path/to/label_hands.txt --tests path/to/tests/
    python3 test_hero_cards.py --limit 50
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from core.config import load_config
from detection.card_detector import detect_hero_cards


# ── label loading ─────────────────────────────────────────────────────────────

def _load_labels(path: str) -> dict[str, tuple[str, str]]:
    """Parse label file → {filename: (card1_str, card2_str)}.

    Each non-comment line: "1.png Ah Ks"
    """
    labels = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            labels[parts[0]] = (parts[1].upper()[0] + parts[1][1:].lower(),
                                 parts[2].upper()[0] + parts[2][1:].lower())
        elif len(parts) == 2:
            # "no_cards" annotation
            labels[parts[0]] = None
    return labels


def _normalise(card_str: str) -> str:
    """Normalise card string: rank upper, suit lower.  "AH" → "Ah"."""
    if len(card_str) < 2:
        return card_str
    return card_str[:-1].upper() + card_str[-1].lower()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hero card detection accuracy test")
    parser.add_argument("--tests",   default="tests/",         help="Directory with PNG test images")
    parser.add_argument("--labels",  default="label_hands.txt", help="Label file path")
    parser.add_argument("--config",  default="regions.json",   help="Regions config path")
    parser.add_argument("--limit",   type=int, default=0,      help="Stop after N images (0 = all)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── collect images ────────────────────────────────────────────────────────
    test_dir = Path(args.tests)
    if not test_dir.is_dir():
        print(f"Test directory not found: {test_dir}")
        sys.exit(1)

    images = sorted(
        test_dir.glob("*.png"),
        key=lambda f: (int(f.stem) if f.stem.isdigit() else float("inf"), f.stem),
    )
    if args.limit:
        images = images[:args.limit]

    # ── load labels (optional) ────────────────────────────────────────────────
    labels: dict = {}
    label_file = Path(args.labels)
    if label_file.exists():
        labels = _load_labels(str(label_file))
        print(f"Loaded {len(labels)} labels from {label_file}")
    else:
        print(f"No label file found at {label_file} — running without ground truth")

    # ── run detection ─────────────────────────────────────────────────────────
    print(f"Testing {len(images)} images from {test_dir}…\n")

    COL_FILE  = 18
    COL_CARDS = 8
    COL_CONF  = 7
    COL_EXP   = 8
    COL_RES   = 6

    hdr = (f"{'Image':<{COL_FILE}}  {'Detected':<{COL_CARDS}}  "
           f"{'Conf':>{COL_CONF}}  {'Expected':<{COL_EXP}}  {'Result':<{COL_RES}}  ms")
    print(hdr)
    print("─" * len(hdr))

    mismatches: list[dict] = []
    errors:     list[dict] = []

    total = detected = correct = wrong = no_cards = 0
    total_ms = 0.0

    for img_path in images:
        total += 1
        label = labels.get(img_path.name)   # tuple(c1,c2) | None | missing

        try:
            img = Image.open(img_path).convert("RGB")
            t0  = time.monotonic()
            res = detect_hero_cards(img, cfg)
            ms  = (time.monotonic() - t0) * 1000
            total_ms += ms
        except Exception as e:
            errors.append({"image": img_path.name, "error": str(e)})
            print(f"{img_path.name:<{COL_FILE}}  {'ERROR':<{COL_CARDS}}  "
                  f"{'':>{COL_CONF}}  {'':>{COL_EXP}}  {str(e)[:40]}")
            continue

        if res is None:
            det_str  = "—"
            conf_str = ""
            result   = ""
        else:
            detected += 1
            c1, c2   = _normalise(res.cards[0].to_str()), _normalise(res.cards[1].to_str())
            det_str  = f"{c1} {c2}"
            conf_str = f"{res.confidence:.2f}"
            result   = ""

        # Compare against label when available
        if label is not None and img_path.name in labels:
            exp_str = f"{label[0]} {label[1]}"
            if res is None:
                result = "MISS"
                mismatches.append({
                    "image":    img_path.name,
                    "expected": exp_str,
                    "got":      "—",
                    "conf":     "",
                    "type":     "missed",
                })
            else:
                c1, c2   = _normalise(res.cards[0].to_str()), _normalise(res.cards[1].to_str())
                exp1, exp2 = _normalise(label[0]), _normalise(label[1])
                if c1 == exp1 and c2 == exp2:
                    result  = "OK"
                    correct += 1
                else:
                    result = "FAIL"
                    wrong  += 1
                    mismatches.append({
                        "image":    img_path.name,
                        "expected": exp_str,
                        "got":      f"{c1} {c2}",
                        "conf":     conf_str,
                        "type":     "wrong",
                    })
        elif img_path.name not in labels:
            exp_str = ""
        else:
            # label is None → annotated as no-cards frame
            exp_str = "no_cards"
            if res is not None:
                result = "FP"
                mismatches.append({
                    "image":    img_path.name,
                    "expected": "no_cards",
                    "got":      det_str,
                    "conf":     conf_str,
                    "type":     "false_positive",
                })
            else:
                no_cards += 1
                result = "OK"
                correct += 1

        print(f"{img_path.name:<{COL_FILE}}  {det_str:<{COL_CARDS}}  "
              f"{conf_str:>{COL_CONF}}  {exp_str:<{COL_EXP}}  "
              f"{result:<{COL_RES}}  {ms:.0f}")

    # ── summary ───────────────────────────────────────────────────────────────
    labeled_count = len([k for k in labels if k in {p.name for p in images}])
    print("─" * len(hdr))
    print(f"\n  Images tested    : {total}")
    print(f"  Cards detected   : {detected}")
    print(f"  Avg latency      : {total_ms / max(total, 1):.0f}ms")
    if labeled_count:
        accuracy = correct / labeled_count * 100
        print(f"  Labeled images   : {labeled_count}")
        print(f"  Correct          : {correct}  ({accuracy:.1f}%)")
        print(f"  Wrong            : {wrong}")
        print(f"  Missed           : {labeled_count - correct - wrong}")

    # ── mismatch table ────────────────────────────────────────────────────────
    if mismatches:
        print(f"\n  MISMATCHES ({len(mismatches)} total)\n")
        cw_img  = max(len("Image"),    max(len(m["image"])    for m in mismatches))
        cw_exp  = max(len("Expected"), max(len(m["expected"]) for m in mismatches))
        cw_got  = max(len("Got"),      max(len(m["got"])      for m in mismatches))
        cw_type = max(len("Type"),     max(len(m["type"])     for m in mismatches))
        cw_conf = max(len("Conf"),     max(len(m["conf"])     for m in mismatches))

        sep = (f"  +-{'-'*cw_img}-+-{'-'*cw_exp}-+-{'-'*cw_got}-+"
               f"-{'-'*cw_conf}-+-{'-'*cw_type}-+")
        print(sep)
        print(f"  | {'Image':<{cw_img}} | {'Expected':<{cw_exp}} | "
              f"{'Got':<{cw_got}} | {'Conf':<{cw_conf}} | {'Type':<{cw_type}} |")
        print(sep)
        for m in mismatches:
            print(f"  | {m['image']:<{cw_img}} | {m['expected']:<{cw_exp}} | "
                  f"{m['got']:<{cw_got}} | {m['conf']:<{cw_conf}} | {m['type']:<{cw_type}} |")
        print(sep)

    if errors:
        print(f"\n  ERRORS ({len(errors)})\n")
        for e in errors:
            print(f"  {e['image']}: {e['error']}")

    print()


if __name__ == "__main__":
    main()
