"""macOS screen capture via Quartz — pure functions, no global state."""
import ctypes
import subprocess
import sys
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import Quartz
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


_sc_counter = 0  # used only by the fallback subprocess path


def get_pokerstars_window() -> Optional[dict]:
    """Find the largest PokerStars game window (excludes lobby).

    Returns dict with keys: x, y, w, h, title — or None if not found.
    Requires pyobjc-framework-Quartz (macOS only).
    """
    if not HAS_QUARTZ:
        print("Quartz not available — cannot find PokerStars window.", file=sys.stderr)
        return None

    wins = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    best, best_area = None, 0
    for w in wins:
        if w.get("kCGWindowOwnerName") != "PokerStars":
            continue
        if "lobby" in (w.get("kCGWindowName") or "").lower():
            continue
        b = w.get("kCGWindowBounds", {})
        ww = int(b.get("Width", 0))
        hh = int(b.get("Height", 0))
        if ww < 400 or hh < 300:
            continue
        area = ww * hh
        if area > best_area:
            best_area = area
            title = w.get("kCGWindowName") or ""
            best = {"x": int(b["X"]), "y": int(b["Y"]), "w": ww, "h": hh, "title": title}
    return best


def screenshot(window_bounds: dict) -> Optional["Image.Image"]:
    """Capture a window region as a PIL Image.

    Args:
        window_bounds: dict with x, y, w, h (pixel coords on screen).

    Returns PIL Image (RGB) or None on failure.
    Tries Quartz direct capture first; falls back to screencapture subprocess.
    """
    if HAS_QUARTZ and HAS_NUMPY:
        img = _capture_quartz(window_bounds)
        if img is not None:
            return img
    return _capture_fallback(window_bounds)


def _capture_quartz(window: dict) -> Optional["Image.Image"]:
    x, y, w, h = window["x"], window["y"], window["w"], window["h"]
    try:
        region = Quartz.CGRectMake(x, y, w, h)
        img_ref = Quartz.CGWindowListCreateImageFromArray(
            region,
            Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
            ),
            Quartz.kCGWindowImageDefault,
        )
        if img_ref is None:
            return None
        width  = Quartz.CGImageGetWidth(img_ref)
        height = Quartz.CGImageGetHeight(img_ref)
        cs  = Quartz.CGColorSpaceCreateDeviceRGB()
        ctx = Quartz.CGBitmapContextCreate(
            None, width, height, 8, width * 4, cs,
            Quartz.kCGImageAlphaPremultipliedLast,
        )
        if ctx is None:
            return None
        Quartz.CGContextDrawImage(ctx, Quartz.CGRectMake(0, 0, width, height), img_ref)
        data_ptr = Quartz.CGBitmapContextGetData(ctx)
        if data_ptr is None:
            return None
        size = width * height * 4
        buf  = (ctypes.c_uint8 * size).from_address(data_ptr)
        arr  = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4)).copy()
        return Image.fromarray(arr[:, :, :3], "RGB")
    except Exception:
        return None


def _capture_fallback(window: dict) -> Optional["Image.Image"]:
    """Slow but reliable fallback using macOS screencapture binary."""
    global _sc_counter
    _sc_counter += 1
    x, y, w, h = window["x"], window["y"], window["w"], window["h"]
    path = f"/tmp/poker_tracker_sc_{_sc_counter}.png"
    try:
        subprocess.run(
            ["screencapture", "-R", f"{x},{y},{w},{h}", path],
            timeout=5, capture_output=True,
        )
        return Image.open(path).convert("RGB")
    except Exception:
        return None
