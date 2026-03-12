#!/usr/bin/env python3
"""Train (or fine-tune) the full-card classifier on hero crops + board crops.

Training data:
  Hero crops  — tests/ directory + label_hands.txt  (same pipeline as train_corner_model.py)
  Board crops — captures/ + label_cards.json          (board slot regions from regions.json)

The resulting model (card_detector_nn.pth) handles both hero and board inputs because
it has seen both kinds of crops during training.

Usage:
  venv/bin/python3 NN/train_card_model.py
  venv/bin/python3 NN/train_card_model.py --hero-only   # skip board crops
  venv/bin/python3 NN/train_card_model.py --epochs 40
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ── Config ────────────────────────────────────────────────────────────────────

TESTS_DIR     = Path("tests")
LABELS_FILE   = Path("label_hands.txt")
CAPTURES_DIR  = Path("captures")
BOARD_LABELS  = Path("label_cards.json")
REGIONS_FILE  = Path("regions.json")
OUTPUT_MODEL  = Path("NN/card_detector_nn.pth")

MODEL_W   = 128
MODEL_H   = 192
EPOCHS    = 60
BATCH     = 32
LR        = 3e-4
VAL_SPLIT = 0.15
SEED      = 42

RANK_MAP    = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]
SUIT_MAP    = ["c","d","h","s"]
RANK_TO_IDX = {r: i for i, r in enumerate(RANK_MAP)}
SUIT_TO_IDX = {s: i for i, s in enumerate(SUIT_MAP)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_card(card_str: str):
    """'Ah' → (rank_idx, suit_idx).  Returns None if invalid."""
    s = card_str.strip()
    if len(s) < 2:
        return None
    rank = s[:-1].upper()
    suit = s[-1].lower()
    if rank not in RANK_TO_IDX or suit not in SUIT_TO_IDX:
        return None
    return RANK_TO_IDX[rank], SUIT_TO_IDX[suit]


def _load_hero_samples(tests_dir: Path, labels_file: Path) -> list:
    """Return list of (PIL Image, rank_idx, suit_idx) from hero card crops."""
    if not labels_file.exists():
        print(f"  Hero labels not found: {labels_file}")
        return []

    samples = []
    skipped = 0
    for line in labels_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        fname = parts[0]
        cards = [parts[1].lower(), parts[2].lower()]
        img_path = tests_dir / fname
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            card_crops = _extract_hero_crops(img)
            for card_img, card_str in zip(card_crops, cards):
                parsed = _parse_card(card_str)
                if parsed is None:
                    continue
                samples.append((card_img, parsed[0], parsed[1]))
        except Exception as e:
            skipped += 1
    print(f"  Hero samples: {len(samples)}  (skipped: {skipped})")
    return samples


def _extract_hero_crops(img: Image.Image) -> list:
    """Split a full screenshot into two hero card images (same as corner model trainer)."""
    w, h = img.size
    hero = img.crop((int(0.38 * w), int(0.55 * h), int(0.58 * w), int(0.75 * h)))
    cw, ch = hero.size
    return [
        hero.crop((cw // 8,     5, cw // 2 - 5, ch - 5)).convert("RGB"),
        hero.crop((cw // 2 + 5, 5, 7 * cw // 8, ch - 5)).convert("RGB"),
    ]


def _load_board_samples(captures_dir: Path, board_labels: Path, regions_file: Path) -> list:
    """Return list of (PIL Image, rank_idx, suit_idx) from board slot crops."""
    if not board_labels.exists():
        print(f"  Board labels not found: {board_labels}")
        return []
    if not regions_file.exists():
        print(f"  Regions config not found: {regions_file}")
        return []

    regions_data = json.loads(regions_file.read_text())
    board_regions = regions_data.get("board", [])
    if not board_regions:
        print("  No board regions in config.")
        return []

    labels = json.loads(board_labels.read_text())
    samples = []
    skipped = 0

    for fname, label in labels.items():
        exp_board = label.get("board")
        if not exp_board:
            continue
        img_path = captures_dir / fname
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size
            for i, card_str in enumerate(exp_board):
                if i >= len(board_regions):
                    break
                r = board_regions[i]
                x0 = int(r[0] * iw)
                y0 = int(r[1] * ih)
                x1 = int(r[2] * iw)
                y1 = int(r[3] * ih)
                slot_crop = img.crop((x0, y0, x1, y1)).convert("RGB")
                parsed = _parse_card(card_str)
                if parsed is None:
                    continue
                samples.append((slot_crop, parsed[0], parsed[1]))
        except Exception as e:
            skipped += 1
            print(f"  Error {fname}: {e}")

    print(f"  Board samples: {len(samples)}  (skipped: {skipped})")
    return samples


# ── Dataset ───────────────────────────────────────────────────────────────────

class CardDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, rank, suit = self.samples[i]
        return self.transform(img), rank, suit


# ── Model ─────────────────────────────────────────────────────────────────────

class DualHeadModel(nn.Module):
    def __init__(self, backbone, rank_head, suit_head):
        super().__init__()
        self.backbone  = backbone
        self.rank_head = rank_head
        self.suit_head = suit_head

    def forward(self, x):
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)


def build_model(pretrained_pth: str | None = None) -> DualHeadModel:
    backbone = models.mobilenet_v2(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    rank_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 13))
    suit_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 4))
    model = DualHeadModel(backbone, rank_head, suit_head)

    if pretrained_pth and Path(pretrained_pth).exists():
        state = torch.load(pretrained_pth, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Loaded weights from {pretrained_pth}")
    else:
        print("  Training from random init")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train full-card classifier")
    parser.add_argument("--epochs",    type=int,  default=EPOCHS)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--hero-only", action="store_true", help="Skip board crops")
    parser.add_argument("--no-pretrain", action="store_true", help="Don't load existing weights")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    train_tf = transforms.Compose([
        transforms.Resize((MODEL_H, MODEL_W)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((MODEL_H, MODEL_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("Loading training data...")
    hero_samples  = _load_hero_samples(TESTS_DIR, LABELS_FILE)
    board_samples = [] if args.hero_only else _load_board_samples(CAPTURES_DIR, BOARD_LABELS, REGIONS_FILE)

    # Oversample board crops to balance with hero samples
    if board_samples and hero_samples:
        repeat = max(1, len(hero_samples) // max(len(board_samples), 1) // 2)
        board_samples = board_samples * repeat
        print(f"  Board samples repeated x{repeat} → {len(board_samples)}")

    all_samples = hero_samples + board_samples
    print(f"  Total samples: {len(all_samples)}")

    if len(all_samples) < 20:
        print("Too few samples — aborting.")
        sys.exit(1)

    random.shuffle(all_samples)
    n_val     = max(1, int(len(all_samples) * VAL_SPLIT))
    val_smp   = all_samples[:n_val]
    train_smp = all_samples[n_val:]

    train_ds = CardDataset(train_smp, train_tf)
    val_ds   = CardDataset(val_smp,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  |  train={len(train_smp)}  val={len(val_smp)}")

    pretrained = None if args.no_pretrain else str(OUTPUT_MODEL)
    model     = build_model(pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_path = str(OUTPUT_MODEL)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = 0.0
        for imgs, ranks, suits in train_dl:
            imgs, ranks, suits = imgs.to(device), ranks.to(device), suits.to(device)
            optimizer.zero_grad()
            rl, sl = model(imgs)
            loss = criterion(rl, ranks) + 2.0 * criterion(sl, suits)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        model.eval()
        r_ok = s_ok = both_ok = total = 0
        with torch.no_grad():
            for imgs, ranks, suits in val_dl:
                imgs, ranks, suits = imgs.to(device), ranks.to(device), suits.to(device)
                rl, sl = model(imgs)
                rp = torch.argmax(rl, 1)
                sp = torch.argmax(sl, 1)
                r_ok    += (rp == ranks).sum().item()
                s_ok    += (sp == suits).sum().item()
                both_ok += ((rp == ranks) & (sp == suits)).sum().item()
                total   += ranks.size(0)

        acc = both_ok / max(total, 1)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={t_loss/len(train_dl):.3f}"
              f"  rank={r_ok/total:.0%}  suit={s_ok/total:.0%}  both={acc:.0%}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    print(f"\nBest val accuracy: {best_acc:.1%}")
    print(f"Model saved → {best_path}")


if __name__ == "__main__":
    main()
