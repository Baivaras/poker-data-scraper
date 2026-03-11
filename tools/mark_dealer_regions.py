#!/usr/bin/env python3
"""
GUI tool for labeling dealer button regions for each seat.

Opens a screenshot with all existing regions shown as reference overlays.
You draw one small bounding box per seat covering where the dealer chip
would appear when that seat holds the button.

Also lets you set the hero_seat number (which seat index is you).

Saves dealer regions back into regions.json under seats[i]["dealer"].

Usage:
    python3 mark_dealer_regions.py tests/1.png
    python3 mark_dealer_regions.py tests/1.png --regions regions.json
"""

import sys
import json
import argparse
import tkinter as tk
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    print("pip install pillow")
    sys.exit(1)

MAX_W, MAX_H = 1440, 860

# Colour for each dealer slot
_DEALER_COLOUR  = "#FFD700"   # gold — matches the chip
_EXISTING_ALPHA = "#555555"   # muted colour for reference overlays

# Existing region keys → display colour (reference only, not editable here)
_REF_REGIONS = [
    ("hero",        "#44AAFF"),
    ("hero_name",   "#88CCFF"),
    ("hero_stack",  "#AADDFF"),
    ("action",      "#FF4444"),
    ("pot",         "#FFCC00"),
    ("hero_dealer", "#FFD700"),
    ("hero_bet",    "#44FFAA"),
]
_REF_SEAT_FIELDS = {
    "name":   "#FF8C00",
    "stack":  "#FFB347",
    "bet":    "#FF6600",
    "avatar": "#CC4400",
    "cards":  "#FF9966",
}


class DealerRegionMarker:
    def __init__(self, root: tk.Tk, img_path: str, regions_file: str):
        self.root          = root
        self.regions_file  = regions_file
        self.root.title(f"Dealer Region Marker  —  {img_path}")
        self.root.configure(bg="#1e1e1e")

        # ── load image ───────────────────────────────────────────────────────
        self.orig          = Image.open(img_path).convert("RGB")
        self.img_w, self.img_h = self.orig.size
        self.scale         = min(MAX_W / self.img_w, MAX_H / self.img_h, 1.0)
        disp_w             = int(self.img_w * self.scale)
        disp_h             = int(self.img_h * self.scale)
        disp_img           = self.orig.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img        = ImageTk.PhotoImage(disp_img)

        # ── state ────────────────────────────────────────────────────────────
        self.active: str | None = None
        self.boxes: dict        = {}        # slot_name → (x0,y0,x1,y1) pixels
        self.canvas_rects: dict = {}
        self.canvas_labels: dict= {}
        self._drag_start        = None
        self._drag_rect         = None
        self._history: list     = []

        # ── load existing regions.json ────────────────────────────────────────
        self._data           = self._load_json()
        self._num_seats      = len(self._data.get("seats", []))
        # Seat dealer slots + two hero-specific slots
        self._slot_names     = (
            [f"SEAT_{i+1}_DEALER" for i in range(self._num_seats)]
            + ["HERO_DEALER", "HERO_BET"]
        )
        # Colours per slot type
        self._slot_colours   = {
            **{f"SEAT_{i+1}_DEALER": _DEALER_COLOUR for i in range(self._num_seats)},
            "HERO_DEALER": "#FFD700",   # same gold — it's still the dealer chip
            "HERO_BET":    "#44FFAA",   # green — hero's bet chips
        }

        # Pre-populate from existing data
        for i, seat in enumerate(self._data.get("seats", [])):
            if "dealer" in seat:
                self.boxes[f"SEAT_{i+1}_DEALER"] = self._frac_to_px(seat["dealer"])
        if "hero_dealer" in self._data:
            self.boxes["HERO_DEALER"] = self._frac_to_px(self._data["hero_dealer"])
        if "hero_bet" in self._data:
            self.boxes["HERO_BET"] = self._frac_to_px(self._data["hero_bet"])

        # ── build UI ─────────────────────────────────────────────────────────
        self._build_sidebar()
        self._build_canvas(disp_w, disp_h)

        # Draw reference regions (existing, muted)
        self._draw_reference_regions()

        # Draw any already-saved dealer boxes
        for name, px in self.boxes.items():
            self._draw_box(name, px, redraw=False)
            if name in self._btns:
                self._mark_done(name)

        # Keyboard shortcuts
        self.root.bind("<Escape>",    lambda e: self._select(None))
        self.root.bind("<Return>",    lambda e: self._save())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Command-z>", lambda e: self._undo())

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = tk.Frame(self.root, bg="#252526", width=260)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Dealer Button Regions", bg="#252526", fg="#cccccc",
                 font=("Helvetica", 12, "bold"), pady=8).pack(fill=tk.X)
        tk.Label(sidebar,
                 text="Draw where the dealer chip appears\nfor each seat.",
                 bg="#252526", fg="#888", font=("Helvetica", 9),
                 justify=tk.LEFT, padx=10).pack(fill=tk.X)
        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X, pady=(6, 0))

        # Action buttons
        acts = tk.Frame(sidebar, bg="#252526")
        acts.pack(fill=tk.X, padx=6, pady=6)
        btn_cfg = dict(relief=tk.FLAT, cursor="hand2",
                       font=("Helvetica", 9, "bold"), pady=4)
        tk.Button(acts, text="↩ Undo", bg="#3a3a3a", fg="#cccccc",
                  command=self._undo, **btn_cfg).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(acts, text="✕ Del", bg="#3a3a3a", fg="#ff8888",
                  command=self._delete_active, **btn_cfg).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        tk.Button(acts, text="⌫ All", bg="#3a3a3a", fg="#ff6644",
                  command=self._clear_all, **btn_cfg).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X)

        # Dealer slot buttons
        self._btns: dict = {}
        slots_frame = tk.Frame(sidebar, bg="#252526")
        slots_frame.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(slots_frame, text="SEAT DEALER REGIONS", bg="#252526", fg="#888",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(0, 4))

        for name in self._slot_names:
            # Section divider before hero-specific slots
            if name == "HERO_DEALER":
                tk.Frame(slots_frame, bg="#444", height=1).pack(fill=tk.X, pady=(6, 4))
                tk.Label(slots_frame, text="HERO REGIONS", bg="#252526", fg="#888",
                         font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(0, 4))

            colour = self._slot_colours[name]
            row = tk.Frame(slots_frame, bg="#252526", pady=3)
            row.pack(fill=tk.X)
            btn = tk.Button(
                row, text=name, bg="#333", fg=colour,
                activebackground=colour, activeforeground="#000",
                font=("Helvetica", 10, "bold"), anchor="w",
                width=16, relief=tk.FLAT, cursor="hand2",
                command=lambda n=name: self._select(n),
            )
            btn.pack(side=tk.LEFT)
            status = tk.Label(row, text="—", bg="#252526", fg="#666",
                              font=("Helvetica", 9), width=4)
            status.pack(side=tk.LEFT, padx=4)
            self._btns[name] = (btn, status)

        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X, pady=(4, 0))

        # hero_seat input
        hero_frame = tk.Frame(sidebar, bg="#252526")
        hero_frame.pack(fill=tk.X, padx=8, pady=8)
        tk.Label(hero_frame, text="HERO SEAT NUMBER", bg="#252526", fg="#888",
                 font=("Helvetica", 8, "bold")).pack(anchor="w", pady=(0, 4))
        hint = tk.Label(hero_frame,
                        text="Which seat index (1–6) is you?",
                        bg="#252526", fg="#666", font=("Helvetica", 8))
        hint.pack(anchor="w")

        entry_row = tk.Frame(hero_frame, bg="#252526")
        entry_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(entry_row, text="hero_seat:", bg="#252526", fg="#cccccc",
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        self._hero_seat_var = tk.StringVar(
            value=str(self._data.get("hero_seat", "")))
        entry = tk.Entry(entry_row, textvariable=self._hero_seat_var,
                         width=4, bg="#3a3a3a", fg="#FFD700",
                         insertbackground="white",
                         font=("Helvetica", 11, "bold"), relief=tk.FLAT)
        entry.pack(side=tk.LEFT, padx=6)

        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X)

        # Legend for reference colours
        legend = tk.Frame(sidebar, bg="#252526")
        legend.pack(fill=tk.X, padx=8, pady=6)
        tk.Label(legend, text="REFERENCE OVERLAYS (existing)", bg="#252526",
                 fg="#555", font=("Helvetica", 8, "bold")).pack(anchor="w")
        for label, colour in [("Hero / action / pot", "#44AAFF"),
                               ("Seat names / stacks", "#FF8C00"),
                               ("Seat bets / cards", "#FF9966")]:
            row = tk.Frame(legend, bg="#252526")
            row.pack(fill=tk.X)
            tk.Label(row, text="■", bg="#252526", fg=colour,
                     font=("Helvetica", 10)).pack(side=tk.LEFT)
            tk.Label(row, text=label, bg="#252526", fg="#555",
                     font=("Helvetica", 8)).pack(side=tk.LEFT, padx=4)

        # Save button
        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X, pady=(4, 0))
        tk.Button(sidebar, text="💾  Save  regions.json",
                  bg="#0e7a0d", fg="white",
                  font=("Helvetica", 11, "bold"), relief=tk.FLAT, pady=8,
                  cursor="hand2", command=self._save).pack(
            fill=tk.X, padx=8, pady=8)
        tk.Label(sidebar,
                 text="Click region → drag on image\nCtrl+Z undo | Esc deselect | Enter save",
                 bg="#252526", fg="#555", font=("Helvetica", 8),
                 justify=tk.LEFT).pack(padx=8, pady=(0, 8))

    # ── canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self, disp_w: int, disp_h: int):
        frame = tk.Frame(self.root, bg="#1e1e1e")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, width=disp_w, height=disp_h,
                                bg="#000", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.status_bar = tk.Label(
            frame, text="Select a seat on the left, then draw where the dealer chip appears.",
            bg="#007acc", fg="white", font=("Helvetica", 10),
            anchor="w", pady=4, padx=8)
        self.status_bar.pack(fill=tk.X)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _draw_reference_regions(self):
        """Draw all existing regions as muted overlays (not interactive)."""
        def _draw(frac, colour, label=""):
            if not frac:
                return
            px = self._frac_to_px(frac)
            cx0, cy0 = px[0] * self.scale, px[1] * self.scale
            cx1, cy1 = px[2] * self.scale, px[3] * self.scale
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                         outline=colour, width=1, dash=(4, 4))
            if label:
                self.canvas.create_text(cx0 + 3, cy0 + 2, anchor=tk.NW,
                                        text=label, fill=colour,
                                        font=("Helvetica", 7))

        for key, colour in _REF_REGIONS:
            _draw(self._data.get(key), colour, key.upper())

        for i, seat in enumerate(self._data.get("seats", [])):
            for field, colour in _REF_SEAT_FIELDS.items():
                if field in seat:
                    _draw(seat[field], colour, f"S{i+1}_{field[:3].upper()}")

    # ── mouse events ─────────────────────────────────────────────────────────

    def _on_press(self, event):
        if not self.active:
            self.status_bar.config(text="⚠  Select a seat button first.")
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect:
            self.canvas.delete(self._drag_rect)

    def _on_drag(self, event):
        if not self._drag_start:
            return
        if self._drag_rect:
            self.canvas.delete(self._drag_rect)
        x0, y0 = self._drag_start
        colour = self._slot_colours.get(self.active, _DEALER_COLOUR)
        self._drag_rect = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=colour, width=2, dash=(6, 3))

    def _on_release(self, event):
        if not self._drag_start or not self.active:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None
        if self._drag_rect:
            self.canvas.delete(self._drag_rect)
            self._drag_rect = None
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return
        px = (int(x0 / self.scale), int(y0 / self.scale),
              int(x1 / self.scale), int(y1 / self.scale))
        self._history.append((self.active, self.boxes.get(self.active)))
        self.boxes[self.active] = px
        self._draw_box(self.active, px, redraw=True)
        self._mark_done(self.active)
        self._advance()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_box(self, name: str, px: tuple, redraw: bool = True):
        colour = self._slot_colours.get(name, _DEALER_COLOUR)
        cx0, cy0 = px[0] * self.scale, px[1] * self.scale
        cx1, cy1 = px[2] * self.scale, px[3] * self.scale
        if redraw:
            if name in self.canvas_rects:
                self.canvas.delete(self.canvas_rects[name])
            if name in self.canvas_labels:
                self.canvas.delete(self.canvas_labels[name])
        rid = self.canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                           outline=colour, width=2)
        tid = self.canvas.create_text(cx0 + 4, cy0 + 4, anchor=tk.NW,
                                      text=name, fill=colour,
                                      font=("Helvetica", 9, "bold"))
        self.canvas_rects[name]  = rid
        self.canvas_labels[name] = tid

    # ── sidebar helpers ───────────────────────────────────────────────────────

    def _select(self, name: str | None):
        self.active = name
        for n, (btn, _) in self._btns.items():
            colour = self._slot_colours.get(n, _DEALER_COLOUR)
            if n == name:
                btn.config(bg=colour, fg="#000")
                hint = ("hero's bet chips on the table" if n == "HERO_BET"
                        else "where the dealer chip appears")
                self.status_bar.config(
                    text=f"Drawing  {name}  — click and drag {hint}",
                    bg=colour, fg="#000")
            else:
                btn.config(bg="#333", fg=colour)
        if name is None:
            self.status_bar.config(
                text="Select a region on the left, then draw on the image.",
                bg="#007acc", fg="white")

    def _mark_done(self, name: str):
        _, status = self._btns[name]
        status.config(text="✓", fg="#44ff88")

    def _advance(self):
        if self.active in self._slot_names:
            idx = self._slot_names.index(self.active)
            for n in self._slot_names[idx + 1:]:
                if n not in self.boxes:
                    self._select(n)
                    return
        self._select(None)
        self.status_bar.config(
            text="All dealer regions marked — press Enter or click Save.",
            bg="#0e7a0d", fg="white")

    def _erase_box(self, name: str):
        if name in self.canvas_rects:
            self.canvas.delete(self.canvas_rects.pop(name))
        if name in self.canvas_labels:
            self.canvas.delete(self.canvas_labels.pop(name))
        if name in self._btns:
            _, status = self._btns[name]
            status.config(text="—", fg="#666")

    def _undo(self):
        if not self._history:
            self.status_bar.config(text="⚠  Nothing to undo.", bg="#555", fg="white")
            return
        name, old_px = self._history.pop()
        self._erase_box(name)
        self.boxes.pop(name, None)
        if old_px is not None:
            self.boxes[name] = old_px
            self._draw_box(name, old_px, redraw=False)
            self._mark_done(name)
        self._select(name)
        self.status_bar.config(text=f"↩  Undid  {name}", bg="#444", fg="white")

    def _delete_active(self):
        if not self.active:
            self.status_bar.config(text="⚠  No region selected.", bg="#555", fg="white")
            return
        if self.active not in self.boxes:
            self.status_bar.config(text=f"⚠  {self.active} not drawn yet.", bg="#555", fg="white")
            return
        self._history.append((self.active, self.boxes.pop(self.active)))
        self._erase_box(self.active)
        self._select(self.active)
        self.status_bar.config(
            text=f"✕  Deleted  {self.active}  — redraw if needed", bg="#444", fg="white")

    def _clear_all(self):
        for name in list(self.boxes.keys()):
            self._history.append((name, self.boxes.pop(name)))
            self._erase_box(name)
        self._select(None)
        self.status_bar.config(text="⌫  Cleared all dealer regions.", bg="#444", fg="white")

    # ── save / load ───────────────────────────────────────────────────────────

    def _px_to_frac(self, px: tuple) -> list:
        x0, y0, x1, y1 = px
        return [round(x0 / self.img_w, 4), round(y0 / self.img_h, 4),
                round(x1 / self.img_w, 4), round(y1 / self.img_h, 4)]

    def _frac_to_px(self, frac: list) -> tuple:
        x0, y0, x1, y1 = frac
        return (int(x0 * self.img_w), int(y0 * self.img_h),
                int(x1 * self.img_w), int(y1 * self.img_h))

    def _load_json(self) -> dict:
        p = Path(self.regions_file)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text())
        except Exception as e:
            print(f"Warning: could not load {self.regions_file}: {e}", file=sys.stderr)
            return {}

    def _save(self):
        # Validate hero_seat
        hero_seat_str = self._hero_seat_var.get().strip()
        hero_seat = None
        if hero_seat_str:
            try:
                hero_seat = int(hero_seat_str)
                if not (1 <= hero_seat <= 6):
                    raise ValueError
            except ValueError:
                self.status_bar.config(
                    text="⚠  hero_seat must be a number between 1 and 6.",
                    bg="#cc4400", fg="white")
                return

        # Merge dealer boxes into existing data
        data = self._data.copy()

        # Ensure seats list is long enough
        seats = list(data.get("seats", []))
        while len(seats) < self._num_seats:
            seats.append({})

        for name, px in self.boxes.items():
            frac = self._px_to_frac(px)
            if name == "HERO_DEALER":
                data["hero_dealer"] = frac
            elif name == "HERO_BET":
                data["hero_bet"] = frac
            else:
                # "SEAT_1_DEALER" → seat index 0
                seat_idx = int(name.split("_")[1]) - 1
                if seat_idx < len(seats):
                    seats[seat_idx]["dealer"] = frac

        data["seats"] = seats

        if hero_seat is not None:
            data["hero_seat"] = hero_seat

        Path(self.regions_file).write_text(json.dumps(data, indent=2))

        n = len(self.boxes)
        msg = f"✓  Saved {n} dealer region(s)"
        if hero_seat is not None:
            msg += f"  +  hero_seat={hero_seat}"
        msg += f"  →  {self.regions_file}"
        self.status_bar.config(text=msg, bg="#0e7a0d", fg="white")
        print(msg)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Label dealer button regions per seat")
    parser.add_argument("image", help="Screenshot PNG to label on")
    parser.add_argument("--regions", default="regions.json",
                        help="regions.json to read/write (default: regions.json)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"File not found: {args.image}")
        sys.exit(1)

    root = tk.Tk()
    DealerRegionMarker(root, args.image, args.regions)
    root.mainloop()


if __name__ == "__main__":
    main()
