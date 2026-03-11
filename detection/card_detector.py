"""Hero card detection: variance gate → NN classifier → structured result."""
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.config import TrackerConfig, crop_region
from core.models import Card, HeroCardsResult

# NN import — path is relative to this file's directory
_NN_DIR = Path(__file__).parent.parent / "NN"   # project_root/NN/
sys.path.insert(0, str(_NN_DIR.parent))          # add project root to path
from NN.nn_card_reader import classify_card

_RANK_MAP = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
_SUIT_MAP = ['c', 'd', 'h', 's']

_hand_counter = 0  # module-level counter for hand_id generation


def _variance(img: Image.Image) -> float:
    arr = np.array(img.convert("L"), dtype=float)
    return float(arr.std())


def _classify_with_confidence(crop: Image.Image, model_path: str) -> tuple[str, float]:
    """Classify a card crop and compute softmax confidence.

    Returns (card_str, confidence) where confidence = min(rank_conf, suit_conf).
    """
    import torch
    import torch.nn as nn
    from torchvision import transforms, models as tv_models

    # Use the cached model from nn_card_reader
    # We re-invoke classify_card but also need logits for confidence.
    # To avoid duplicating model loading, we reach into the module's cache.
    import NN.nn_card_reader as _nn
    model, device = _nn._load_model(model_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    card = crop.convert("RGB").resize((128, 192), Image.LANCZOS)
    tensor = transform(card).unsqueeze(0).to(device)
    with torch.no_grad():
        rank_logits, suit_logits = model(tensor)
        rank_probs = torch.softmax(rank_logits, dim=1)
        suit_probs = torch.softmax(suit_logits, dim=1)
        rank_idx   = torch.argmax(rank_probs, dim=1).item()
        suit_idx   = torch.argmax(suit_probs, dim=1).item()
        rank_conf  = rank_probs[0, rank_idx].item()
        suit_conf  = suit_probs[0, suit_idx].item()

    card_str   = f"{_RANK_MAP[rank_idx]}{_SUIT_MAP[suit_idx]}"
    confidence = rank_conf * suit_conf  # joint probability
    return card_str, confidence


def detect_hero_cards(img: Image.Image, cfg: TrackerConfig) -> Optional[HeroCardsResult]:
    """Detect and classify the hero's hole cards in a screenshot.

    Pipeline:
      1. Crop hero region from full screenshot.
      2. Variance gate — if std < threshold, no cards present → return None.
      3. Split crop into left/right halves → classify each via NN.
      4. Return HeroCardsResult (or None if confidence too low).

    Args:
        img: Full screenshot (PIL Image, RGB).
        cfg: TrackerConfig with region + threshold settings.

    Returns HeroCardsResult or None.
    """
    global _hand_counter

    # 1. Fast variance gate on calibrated hero region (~1ms)
    gate_crop = crop_region(img, cfg.regions.hero)
    if _variance(gate_crop) < cfg.card_variance_threshold:
        return None

    # 2. Crop the NN classification region (matches model training fractions)
    nn_crop = crop_region(img, cfg.regions.hero_nn_region())

    # 3. Split into left / right card using same insets as the original trainer
    cw, ch = nn_crop.size
    left_crop  = nn_crop.crop((cw // 8,     5, cw // 2 - 5, ch - 5))
    right_crop = nn_crop.crop((cw // 2 + 5, 5, 7 * cw // 8, ch - 5))

    model_path = str(Path(__file__).parent.parent / cfg.model_path)

    try:
        card1_str, conf1 = _classify_with_confidence(left_crop,  model_path)
        card2_str, conf2 = _classify_with_confidence(right_crop, model_path)
    except Exception as e:
        print(f"[card_detector] NN error: {e}", file=sys.stderr)
        return None

    # Per-card confidence: product of rank and suit softmax peaks.
    # Overall = min of two cards (weakest-link principle).
    confidence = min(conf1, conf2)

    if confidence < cfg.confidence_threshold:
        print(
            f"[card_detector] Low confidence ({confidence:.2f}): {card1_str} {card2_str} — skipping",
            file=sys.stderr,
        )
        return None

    def _parse(s: str) -> Card:
        return Card(rank=s[:-1], suit=s[-1])

    _hand_counter += 1
    return HeroCardsResult(
        cards=[_parse(card1_str), _parse(card2_str)],
        confidence=round(confidence, 4),
        timestamp=int(time.time()),
        hand_id=f"hand_{_hand_counter:04d}",
    )
