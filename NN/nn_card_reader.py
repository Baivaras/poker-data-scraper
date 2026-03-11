#!/usr/bin/env python3
"""Neural network card classifier."""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

_RANK_MAP = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
_SUIT_MAP = ['c', 'd', 'h', 's']

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None
_device = None
_loaded_path = None


class _DualHeadModel(nn.Module):
    def __init__(self, backbone, rank_head, suit_head):
        super().__init__()
        self.backbone = backbone
        self.rank_head = rank_head
        self.suit_head = suit_head

    def forward(self, x):
        features = self.backbone(x)
        return self.rank_head(features), self.suit_head(features)


def _load_model(model_path: str):
    global _model, _device, _loaded_path
    if _model is not None and _loaded_path == model_path:
        return _model, _device
    backbone = models.mobilenet_v2(weights=None)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    rank_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 13))
    suit_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 4))
    model = _DualHeadModel(backbone, rank_head, suit_head)
    _device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=_device))
    model.to(_device)
    model.eval()
    _model = model
    _loaded_path = model_path
    return _model, _device


def classify_card(card_img: Image.Image, model_path: str = 'card_detector_nn.pth',
                  size: tuple = (128, 192)) -> str:
    """Classify a single pre-cropped card image → e.g. 'Ah', 'Td', 'Kc'.

    Args:
        card_img:   Any PIL image of the card (or card corner).
        model_path: Path to the .pth weights file.
        size:       (width, height) to resize to before inference.
                    Use (128, 192) for full-card hero model,
                    (64, 96) for corner-based board model.
    """
    model, device = _load_model(model_path)
    card = card_img.convert('RGB').resize(size, Image.LANCZOS)
    tensor = _transform(card).unsqueeze(0).to(device)
    with torch.no_grad():
        rank_logits, suit_logits = model(tensor)
        rank = torch.argmax(rank_logits, dim=1).item()
        suit = torch.argmax(suit_logits, dim=1).item()
    return f"{_RANK_MAP[rank]}{_SUIT_MAP[suit]}"


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python nn_card_reader.py <card_image>')
        sys.exit(1)
    model_path = str(Path(__file__).parent / 'card_detector_nn.pth')
    result = classify_card(Image.open(sys.argv[1]), model_path)
    print(result)
