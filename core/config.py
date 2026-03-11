"""Configuration dataclasses and loader for the poker tracker."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


@dataclass
class RegionConfig:
    # hero: calibrated region used for the fast variance gate check
    hero: List[float]               = field(default_factory=lambda: [0.38, 0.55, 0.58, 0.75])
    # hero_nn: region fed to the NN classifier (matches training data fractions)
    hero_nn: Optional[List[float]]  = None
    hero_name: Optional[List[float]] = None   # hero username label
    hero_stack: Optional[List[float]] = None  # hero chip stack
    hero_dealer: Optional[List[float]] = None # where dealer chip appears when hero has BTN
    hero_bet: Optional[List[float]] = None    # hero's bet chips on table
    board: List[List[float]]        = field(default_factory=list)
    action: List[float]             = field(default_factory=lambda: [0.20, 0.80, 0.80, 0.96])
    pot: List[float]                = field(default_factory=lambda: [0.35, 0.27, 0.65, 0.35])
    seats: List[dict]               = field(default_factory=list)
    hero_seat: Optional[int]        = None    # 1-based seat number hero occupies (optional)

    def hero_nn_region(self) -> List[float]:
        """Return the NN classification region (falls back to hero if not set)."""
        return self.hero_nn if self.hero_nn is not None else self.hero


@dataclass
class TrackerConfig:
    regions: RegionConfig           = field(default_factory=RegionConfig)
    card_variance_threshold: float  = 40.0
    debounce_frames: int            = 2
    confidence_threshold: float     = 0.30
    poll_ms: int                    = 150
    model_path: str                 = "NN/card_detector_nn.pth"
    dealer_match_threshold: float   = 0.55   # min NCC score to accept a dealer match
    dealer_scales: List[float]      = field(default_factory=lambda: [0.7, 0.85, 1.0, 1.15])


def load_config(path: str = "regions.json") -> TrackerConfig:
    cfg = TrackerConfig()
    p = Path(path)
    if not p.exists():
        return cfg
    try:
        data = json.loads(p.read_text())
        r = cfg.regions
        if "hero" in data:
            r.hero = list(data["hero"])
        if "hero_nn" in data:
            r.hero_nn = list(data["hero_nn"])
        if "board" in data:
            r.board = [list(slot) for slot in data["board"]]
        if "action" in data:
            r.action = list(data["action"])
        if "pot" in data:
            r.pot = list(data["pot"])
        if "seats" in data:
            r.seats = data["seats"]
        for key in ("hero_name", "hero_stack", "hero_dealer", "hero_bet"):
            if key in data:
                setattr(r, key, list(data[key]))
        if "hero_seat" in data:
            r.hero_seat = int(data["hero_seat"])
    except Exception as e:
        import sys
        print(f"Warning: could not load {path}: {e}", file=sys.stderr)
    return cfg


def crop_region(img: Image.Image, region: List[float], window_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Crop a fractional region [x0,y0,x1,y1] from img.

    If window_size is given, coords are computed relative to that size.
    Otherwise img.size is used.
    """
    w, h = window_size if window_size else img.size
    x0 = int(region[0] * w)
    y0 = int(region[1] * h)
    x1 = int(region[2] * w)
    y1 = int(region[3] * h)
    # Clamp to image bounds
    x0 = max(0, min(x0, img.width))
    y0 = max(0, min(y0, img.height))
    x1 = max(x0 + 1, min(x1, img.width))
    y1 = max(y0 + 1, min(y1, img.height))
    return img.crop((x0, y0, x1, y1))
