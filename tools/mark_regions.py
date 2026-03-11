#!/usr/bin/env python3
"""Interactive region marker for poker-tracker.

Opens a screenshot and lets you draw bounding boxes for each detection region.
Saves results to regions.json (project root) which all tracker modules load.

Usage:
    python3 tools/mark_regions.py tests/1.png
    python3 tools/mark_regions.py tests/1.png --regions regions.json
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

# Default regions file path (relative to CWD — run from project root)
DEFAULT_REGIONS_FILE = "regions.json"

# Regions to mark, in order.  (name, colour, description)
REGION_DEFS = [
    # ── table / hero ──────────────────────────────────────────────────────────
    ("ACTION",        "#FF4444", "Action buttons  (Fold / Call / Raise)"),
    ("HERO",          "#44AAFF", "Hero hole cards  (variance gate region)"),
    ("HERO_NAME",     "#88CCFF", "Hero: username label"),
    ("HERO_STACK",    "#AADDFF", "Hero: chip stack amount"),
    ("POT",           "#FFCC00", "Pot amount text"),
    # ── board ─────────────────────────────────────────────────────────────────
    ("BOARD_1",       "#44FF88", "Board card 1  (flop)"),
    ("BOARD_2",       "#44FF88", "Board card 2  (flop)"),
    ("BOARD_3",       "#44FF88", "Board card 3  (flop)"),
    ("BOARD_4",       "#AAFFAA", "Board card 4  (turn)"),
    ("BOARD_5",       "#AAFFAA", "Board card 5  (river)"),
    # ── opponent seats ────────────────────────────────────────────────────────
    ("SEAT_1_NAME",   "#FF8C00", "Seat 1: username / action area"),
    ("SEAT_1_STACK",  "#FFB347", "Seat 1: chip stack amount"),
    ("SEAT_1_BET",    "#FF6600", "Seat 1: bet chips on table"),
    ("SEAT_1_AVATAR", "#CC4400", "Seat 1: avatar (dimmed = folded)"),
    ("SEAT_1_CARDS",  "#FF9966", "Seat 1: face-down hole cards"),
    ("SEAT_2_NAME",   "#FF8C00", "Seat 2: username / action area"),
    ("SEAT_2_STACK",  "#FFB347", "Seat 2: chip stack amount"),
    ("SEAT_2_BET",    "#FF6600", "Seat 2: bet chips on table"),
    ("SEAT_2_AVATAR", "#CC4400", "Seat 2: avatar (dimmed = folded)"),
    ("SEAT_2_CARDS",  "#FF9966", "Seat 2: face-down hole cards"),
    ("SEAT_3_NAME",   "#FF8C00", "Seat 3: username / action area"),
    ("SEAT_3_STACK",  "#FFB347", "Seat 3: chip stack amount"),
    ("SEAT_3_BET",    "#FF6600", "Seat 3: bet chips on table"),
    ("SEAT_3_AVATAR", "#CC4400", "Seat 3: avatar (dimmed = folded)"),
    ("SEAT_3_CARDS",  "#FF9966", "Seat 3: face-down hole cards"),
    ("SEAT_4_NAME",   "#FF8C00", "Seat 4: username / action area"),
    ("SEAT_4_STACK",  "#FFB347", "Seat 4: chip stack amount"),
    ("SEAT_4_BET",    "#FF6600", "Seat 4: bet chips on table"),
    ("SEAT_4_AVATAR", "#CC4400", "Seat 4: avatar (dimmed = folded)"),
    ("SEAT_4_CARDS",  "#FF9966", "Seat 4: face-down hole cards"),
    ("SEAT_5_NAME",   "#FF8C00", "Seat 5: username / action area"),
    ("SEAT_5_STACK",  "#FFB347", "Seat 5: chip stack amount"),
    ("SEAT_5_BET",    "#FF6600", "Seat 5: bet chips on table"),
    ("SEAT_5_AVATAR", "#CC4400", "Seat 5: avatar (dimmed = folded)"),
    ("SEAT_5_CARDS",  "#FF9966", "Seat 5: face-down hole cards"),
]

MAX_W, MAX_H = 1440, 860   # max canvas size


class RegionMarker:
    def __init__(self, root: tk.Tk, img_path: str, regions_file: str):
        self.root         = root
        self.regions_file = regions_file
        self.root.title(f"Region Marker  —  {img_path}")
        self.root.configure(bg="#1e1e1e")

        # ── load image ───────────────────────────────────────────────────────
        self.orig = Image.open(img_path).convert("RGB")
        self.img_w, self.img_h = self.orig.size
        scale = min(MAX_W / self.img_w, MAX_H / self.img_h, 1.0)
        self.scale = scale
        disp_w = int(self.img_w * scale)
        disp_h = int(self.img_h * scale)
        disp_img = self.orig.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(disp_img)

        # ── state ────────────────────────────────────────────────────────────
        self.active: str | None  = None
        self.boxes: dict         = {}
        self.canvas_rects: dict  = {}
        self.canvas_labels: dict = {}
        self._drag_start         = None
        self._drag_rect          = None
        self._history: list      = []

        # ── load existing regions.json ────────────────────────────────────────
        existing = self._load_existing()

        # ── layout ───────────────────────────────────────────────────────────
        self._build_sidebar()
        self._build_canvas(disp_w, disp_h)

        for name, frac in existing.items():
            px = self._frac_to_px(frac)
            self.boxes[name] = px
            self._draw_box(name, px, redraw=False)
            if name in self._btns:
                self._mark_done(name)

        self.root.bind("<Escape>",    lambda e: self._select(None))
        self.root.bind("<Return>",    lambda e: self._save())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Command-z>", lambda e: self._undo())

    # ── sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = tk.Frame(self.root, bg="#252526", width=240)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Regions", bg="#252526", fg="#cccccc",
                 font=("Helvetica", 13, "bold"), pady=8).pack(fill=tk.X)
        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X)

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

        list_outer = tk.Frame(sidebar, bg="#252526")
        list_outer.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(list_outer, bg="#252526",
                                        highlightthickness=0)
        scrollbar = tk.Scrollbar(list_outer, orient=tk.VERTICAL,
                                 command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = tk.Frame(self._scroll_canvas, bg="#252526")
        inner_id = self._scroll_canvas.create_window(
            (0, 0), window=inner, anchor=tk.NW)

        def _on_inner_configure(e):
            self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_configure(e):
            self._scroll_canvas.itemconfig(inner_id, width=e.width)

        inner.bind("<Configure>", _on_inner_configure)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_wheel(e):
            self._scroll_canvas.yview_scroll(int(-1 * e.delta), "units")

        self._scroll_canvas.bind("<MouseWheel>", _on_wheel)
        inner.bind("<MouseWheel>", _on_wheel)

        self._btns: dict = {}
        for name, colour, desc in REGION_DEFS:
            row = tk.Frame(inner, bg="#252526", pady=2)
            row.pack(fill=tk.X, padx=8)
            row.bind("<MouseWheel>", _on_wheel)
            btn = tk.Button(
                row, text=name, bg="#333", fg=colour,
                activebackground=colour, activeforeground="#000",
                font=("Helvetica", 10, "bold"), anchor="w",
                width=14, relief=tk.FLAT, cursor="hand2",
                command=lambda n=name: self._select(n),
            )
            btn.pack(side=tk.LEFT)
            btn.bind("<MouseWheel>", _on_wheel)
            status = tk.Label(row, text="—", bg="#252526", fg="#666",
                              font=("Helvetica", 9), width=4)
            status.pack(side=tk.LEFT, padx=4)
            status.bind("<MouseWheel>", _on_wheel)
            desc_lbl = tk.Label(inner, text=desc, bg="#252526", fg="#888",
                                font=("Helvetica", 8), anchor="w")
            desc_lbl.pack(fill=tk.X, padx=12)
            desc_lbl.bind("<MouseWheel>", _on_wheel)
            self._btns[name] = (btn, status)

        tk.Frame(sidebar, bg="#444", height=1).pack(fill=tk.X)
        tk.Button(
            sidebar, text="Save  regions.json", bg="#0e7a0d", fg="white",
            font=("Helvetica", 11, "bold"), relief=tk.FLAT, pady=8,
            cursor="hand2", command=self._save,
        ).pack(fill=tk.X, padx=8, pady=8)
        tk.Label(sidebar,
                 text="Click region → drag on image\nCtrl+Z undo | Esc deselect | Enter save",
                 bg="#252526", fg="#555", font=("Helvetica", 8),
                 justify=tk.LEFT).pack(padx=8, pady=(0, 8))

    # ── canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self, disp_w: int, disp_h: int):
        frame = tk.Frame(self.root, bg="#1e1e1e")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            frame, width=disp_w, height=disp_h,
            bg="#000", cursor="crosshair", highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.status_bar = tk.Label(
            frame, text="Select a region on the left, then draw on the image.",
            bg="#007acc", fg="white", font=("Helvetica", 10), anchor="w",
            pady=4, padx=8,
        )
        self.status_bar.pack(fill=tk.X)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    # ── mouse events ──────────────────────────────────────────────────────────

    def _on_press(self, event):
        if not self.active:
            self.status_bar.config(text="Select a region button first.")
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect:
            self.canvas.delete(self._drag_rect)

    def _on_drag(self, event):
        if not self._drag_start:
            return
        colour = self._colour(self.active)
        if self._drag_rect:
            self.canvas.delete(self._drag_rect)
        x0, y0 = self._drag_start
        self._drag_rect = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline=colour, width=2, dash=(6, 3),
        )

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
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
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
        colour = self._colour(name)
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
            colour = self._colour(n)
            if n == name:
                btn.config(bg=colour, fg="#000")
                self.status_bar.config(
                    text=f"Drawing  {name}  — click and drag on the image",
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
        names = [n for n, _, __ in REGION_DEFS]
        if self.active in names:
            idx = names.index(self.active)
            for n in names[idx + 1:]:
                if n not in self.boxes:
                    self._select(n)
                    return
        self._select(None)
        self.status_bar.config(
            text="All regions marked — press Enter or click Save.",
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
            self.status_bar.config(text="Nothing to undo.", bg="#555", fg="white")
            return
        name, old_px = self._history.pop()
        self._erase_box(name)
        self.boxes.pop(name, None)
        if old_px is not None:
            self.boxes[name] = old_px
            self._draw_box(name, old_px, redraw=False)
            self._mark_done(name)
        self._select(name)
        self.status_bar.config(text=f"Undid  {name}", bg="#444", fg="white")

    def _delete_active(self):
        if not self.active:
            self.status_bar.config(text="No region selected.", bg="#555", fg="white")
            return
        name = self.active
        if name not in self.boxes:
            self.status_bar.config(text=f"{name} not drawn yet.", bg="#555", fg="white")
            return
        self._history.append((name, self.boxes.pop(name)))
        self._erase_box(name)
        self._select(name)
        self.status_bar.config(
            text=f"Deleted  {name}  — redraw if needed", bg="#444", fg="white")

    def _clear_all(self):
        for name in list(self.boxes.keys()):
            self._history.append((name, self.boxes.pop(name)))
            self._erase_box(name)
        self._select(None)
        self.status_bar.config(text="Cleared all regions.", bg="#444", fg="white")

    def _colour(self, name: str) -> str:
        for n, c, _ in REGION_DEFS:
            if n == name:
                return c
        return "#ffffff"

    # ── save / load ───────────────────────────────────────────────────────────

    def _px_to_frac(self, px: tuple) -> list:
        x0, y0, x1, y1 = px
        return [
            round(x0 / self.img_w, 4),
            round(y0 / self.img_h, 4),
            round(x1 / self.img_w, 4),
            round(y1 / self.img_h, 4),
        ]

    def _frac_to_px(self, frac: list) -> tuple:
        x0, y0, x1, y1 = frac
        return (
            int(x0 * self.img_w), int(y0 * self.img_h),
            int(x1 * self.img_w), int(y1 * self.img_h),
        )

    def _save(self):
        if not self.boxes:
            self.status_bar.config(text="Nothing to save yet.", bg="#cc4400", fg="white")
            return

        # Load existing to preserve fields we don't manage (hero_nn, dealer regions, etc.)
        try:
            existing_raw = json.loads(Path(self.regions_file).read_text())
        except Exception:
            existing_raw = {}

        board: list = []
        seats: list = []

        for name, px in self.boxes.items():
            frac = self._px_to_frac(px)
            if name.startswith("BOARD_"):
                idx = int(name.split("_")[1]) - 1
                while len(board) <= idx:
                    board.append(None)
                board[idx] = frac
            elif name.startswith("SEAT_"):
                parts    = name.split("_")   # ["SEAT", "1", "NAME"]
                seat_idx = int(parts[1]) - 1
                field    = parts[2].lower()
                while len(seats) <= seat_idx:
                    seats.append({})
                seats[seat_idx][field] = frac
            elif name == "HERO_NAME":
                existing_raw["hero_name"] = frac
            elif name == "HERO_STACK":
                existing_raw["hero_stack"] = frac
            else:
                existing_raw[name.lower()] = frac

        if any(b is not None for b in board):
            existing_raw["board"] = [b for b in board if b is not None]

        if any(seats):
            # Merge drawn seat fields into existing seat entries (preserve dealer etc.)
            existing_seats = existing_raw.get("seats", [{}] * len(seats))
            while len(existing_seats) < len(seats):
                existing_seats.append({})
            for i, drawn in enumerate(seats):
                existing_seats[i].update(drawn)
            existing_raw["seats"] = existing_seats

        Path(self.regions_file).write_text(json.dumps(existing_raw, indent=2))
        n = len(self.boxes)
        self.status_bar.config(
            text=f"Saved {n} regions to {self.regions_file}",
            bg="#0e7a0d", fg="white")
        print(f"Saved {n} regions to {self.regions_file}")

    def _load_existing(self) -> dict:
        p = Path(self.regions_file)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text())
            out  = {}
            for key, val in data.items():
                if key == "board":
                    for i, frac in enumerate(val):
                        out[f"BOARD_{i+1}"] = frac
                elif key == "seats":
                    for i, seat in enumerate(val):
                        seat_num = i + 1
                        for field in ("name", "stack", "bet", "avatar", "cards"):
                            if field in seat:
                                out[f"SEAT_{seat_num}_{field.upper()}"] = seat[field]
                elif key == "hero_name":
                    out["HERO_NAME"] = val
                elif key == "hero_stack":
                    out["HERO_STACK"] = val
                else:
                    out[key.upper()] = val
            return out
        except Exception:
            return {}


def main():
    parser = argparse.ArgumentParser(description="Interactive region marker")
    parser.add_argument("image", help="Screenshot PNG to use as background")
    parser.add_argument("--regions", default=DEFAULT_REGIONS_FILE,
                        help=f"Path to regions.json (default: {DEFAULT_REGIONS_FILE})")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"File not found: {args.image}")
        sys.exit(1)

    root = tk.Tk()
    RegionMarker(root, args.image, args.regions)
    root.mainloop()


if __name__ == "__main__":
    main()
