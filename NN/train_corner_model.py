#!/usr/bin/env python3
"""
Train a corner-based card classifier using tests/ images + label_hands.txt.

Extracts the top-left corner (rank letter + suit symbol) from each card face,
resizes to CORNER_W x CORNER_H, and trains a DualHead MobileNetV2.
The resulting model is used for community/board card detection.

Usage:
  venv/bin/python3 NN/train_corner_model.py
"""

import sys
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ── Config ────────────────────────────────────────────────────────────────────

TESTS_DIR    = Path("tests")
LABELS_FILE  = Path("label_hands.txt")
OUTPUT_MODEL = Path("NN/corner_card_nn.pth")

CORNER_W  = 74   # matches hero face width
CORNER_H  = 82   # matches hero face height
EPOCHS    = 60
BATCH     = 32
LR        = 5e-4
VAL_SPLIT = 0.15
SEED      = 42

RANK_MAP     = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]
SUIT_MAP     = ["c","d","h","s"]
RANK_TO_IDX  = {r: i for i, r in enumerate(RANK_MAP)}
SUIT_TO_IDX  = {s: i for i, s in enumerate(SUIT_MAP)}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_labels(path: Path) -> dict:
    """Return {filename: [card1_str, card2_str]} for confirmed (non-#) lines."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            out[parts[0]] = [parts[1].lower(), parts[2].lower()]
    return out


def extract_card_crops(img: Image.Image) -> list:
    """Split a full screenshot into two hero card images."""
    w, h = img.size
    hero = img.crop((int(0.38 * w), int(0.55 * h), int(0.58 * w), int(0.75 * h)))
    cw, ch = hero.size
    return [
        hero.crop((cw // 8,     5, cw // 2 - 5, ch - 5)).convert("RGB"),
        hero.crop((cw // 2 + 5, 5, 7 * cw // 8, ch - 5)).convert("RGB"),
    ]


def extract_corner(card_img: Image.Image) -> Image.Image | None:
    """Find the white card face, return top-left corner (rank + suit area)."""
    arr  = np.array(card_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, wmask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(wmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cx, cy, cw, ch = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    face = arr[cy:cy + ch, cx:cx + cw]
    if face.size == 0 or cw < 15 or ch < 15:
        return None
    crop_h = min(CORNER_H, face.shape[0])
    crop_w = min(CORNER_W, face.shape[1])
    return Image.fromarray(face[:crop_h, :crop_w])

# ── Dataset ───────────────────────────────────────────────────────────────────

class CornerDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples   # list of (PIL Image, rank_idx, suit_idx)
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


def build_model(pretrained_pth: str | None = None):
    """Build DualHeadModel.

    If pretrained_pth is given, bootstrap the backbone weights from an
    existing card_detector_nn.pth (avoids downloading from the internet).
    Otherwise the backbone starts with random weights.
    """
    backbone = models.mobilenet_v2(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    rank_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 13))
    suit_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 4))
    model = DualHeadModel(backbone, rank_head, suit_head)

    if pretrained_pth and Path(pretrained_pth).exists():
        # Load weights from existing model — reuses the backbone features
        state = torch.load(pretrained_pth, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Bootstrapped from {pretrained_pth}")

    return model

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    labels = parse_labels(LABELS_FILE)
    print(f"Labels loaded: {len(labels)} hands")

    train_tf = transforms.Compose([
        transforms.Resize((CORNER_H, CORNER_W)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((CORNER_H, CORNER_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Build sample list
    samples = []
    skipped = 0
    for filename, cards in labels.items():
        img_path = TESTS_DIR / filename
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            card_crops = extract_card_crops(img)
            for card_img, card_str in zip(card_crops, cards):
                corner = extract_corner(card_img)
                if corner is None:
                    continue
                rank_str = card_str[0].upper()
                suit_str = card_str[1].lower()
                if rank_str not in RANK_TO_IDX or suit_str not in SUIT_TO_IDX:
                    continue
                samples.append((corner, RANK_TO_IDX[rank_str], SUIT_TO_IDX[suit_str]))
        except Exception as e:
            print(f"  Error {filename}: {e}")
            skipped += 1

    print(f"Corner samples: {len(samples)}  (skipped: {skipped})")
    if len(samples) < 20:
        print("Too few samples — aborting.")
        sys.exit(1)

    # Train / val split
    random.shuffle(samples)
    n_val    = max(1, int(len(samples) * VAL_SPLIT))
    val_smp  = samples[:n_val]
    train_smp = samples[n_val:]

    train_ds = CornerDataset(train_smp, train_tf)
    val_ds   = CornerDataset(val_smp,   val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  |  train={len(train_smp)}  val={len(val_smp)}")

    pretrained = str(Path("NN/card_detector_nn.pth"))
    model     = build_model(pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────────────
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

        # ── validate ──────────────────────────────────────────────────────────
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
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={t_loss/len(train_dl):.3f}"
              f"  rank={r_ok/total:.0%}  suit={s_ok/total:.0%}  both={acc:.0%}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUTPUT_MODEL)

    print(f"\nBest val accuracy: {best_acc:.1%}")
    print(f"Model saved → {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
