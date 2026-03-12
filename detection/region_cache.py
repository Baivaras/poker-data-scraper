"""Lightweight per-region image hash cache.

Lets any detector skip work when a screen region's pixels are unchanged
from the previous frame.  Stack and bet regions are intentionally excluded
from caching — callers simply don't pass those regions here.

Usage:
    cache = RegionCache()

    if cache.changed(img, region):
        # pixels changed — run the real detection
        result = expensive_detection(img, region)
    # else: skip — region looks identical to last frame

    cache.reset()   # call between hands / on reset
"""
from PIL import Image
from core.config import crop_region


def _region_hash(img: Image.Image, region: list) -> bytes:
    """8×8 grayscale thumbnail of the region — fast perceptual hash."""
    return (
        crop_region(img, region)
        .convert("L")
        .resize((8, 8), Image.BILINEAR)
        .tobytes()
    )


class RegionCache:
    """Tracks per-region hashes and reports whether content has changed."""

    def __init__(self):
        self._hashes: dict = {}

    def changed(self, img: Image.Image, region: list) -> bool:
        """Return True if the region's pixels differ from the last call.

        Always returns True on the first call for a new region so the
        caller always runs detection at least once.
        """
        key  = tuple(region)
        h    = _region_hash(img, region)
        prev = self._hashes.get(key)
        self._hashes[key] = h
        return prev != h

    def reset(self):
        """Clear all stored hashes (call on hand reset / street change if needed)."""
        self._hashes.clear()
