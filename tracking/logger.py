"""PokerLogger — terminal + file logging for all tracker events."""
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

from core.hand_state import HandState, PlayerAction, Street
from core.models import HeroCardsResult
from core.table_state import TableState


# ── ANSI ──────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"

_COLOUR_CODES = {
    "green":  _GREEN,
    "yellow": _YELLOW,
    "cyan":   _CYAN,
    "red":    _RED,
    "bold":   _BOLD,
    "dim":    _DIM,
}

# Hearts / diamonds = red;  spades / clubs = white
_SUIT_COLOUR = {"h": _RED, "d": _RED, "s": _WHITE, "c": _WHITE}

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


# ── Tee writer (terminal + log file) ──────────────────────────────────────────

class _TeeWriter:
    def __init__(self, stream, log_file):
        self._stream   = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(_ANSI_RE.sub("", data))
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ── PokerLogger ───────────────────────────────────────────────────────────────

class PokerLogger:
    """Formats and prints all tracker events to the terminal (and optionally a log file).

    Usage:
        logger = PokerLogger(verbose=True)
        logger.setup_log_file()   # optional — call before any output

        tracker = GameTracker(
            ...
            on_hand_start    = logger.on_hand_start,
            on_street_change = logger.on_street_change,
            on_player_action = logger.on_player_action,
            on_your_turn     = logger.on_your_turn,
            on_hand_complete = logger.on_hand_complete,
        )
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    # ── log file ──────────────────────────────────────────────────────────────

    def setup_log_file(self) -> Path:
        """Redirect stdout/stderr through a tee so output goes to terminal AND log file."""
        log_dir  = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"tracker_{time.strftime('%Y%m%d_%H%M%S')}.log"
        log_fh   = log_path.open("w", encoding="utf-8", buffering=1)
        sys.stdout = _TeeWriter(sys.stdout, log_fh)
        sys.stderr = _TeeWriter(sys.stderr, log_fh)
        return log_path

    # ── event handlers ────────────────────────────────────────────────────────

    def on_hand_start(self, result: HeroCardsResult, table: TableState, scan_ms: int = 0):
        dealer = ("hero" if table.dealer_seat == 0
                  else f"seat {table.dealer_seat}"
                  if table.dealer_seat is not None else "?")
        print(f"\n  {'━'*66}", flush=True)
        self._log("HAND START", f"dealer={dealer}  pos={table.hero_position or '?'}  "
                  f"hand={table.hand_id}  {_DIM}scan={scan_ms}ms{_RESET}", "bold")

        cards_str = self._fmt_cards([c.to_str() for c in result.cards])
        print(f"  {self._ts()}  {_BOLD}{'HERO CARDS':<16}{_RESET}  "
              f"{cards_str}  {_DIM}conf={result.confidence:.2f}{_RESET}", flush=True)

        active = [p for p in table.players if p.status == "ACTIVE"]
        for p in active:
            seat = "hero" if p.is_hero else f"seat {p.seat}"
            self._log("", f"  {seat:<8}  {(p.username or ''):22}  "
                      f"stack={p.stack or '?'}  pos={p.position or '?'}", "dim")

    def on_street_change(self, street: Street, board_cards: list = None):
        colours = {"PREFLOP": "dim", "FLOP": "cyan", "TURN": "yellow", "RIVER": "red"}
        colour  = colours.get(street, "")
        code    = _COLOUR_CODES.get(colour, "")
        sep     = f"  {self._ts()}  {code}── {street:<12}{_RESET}  "
        if board_cards:
            print(f"{sep}{self._fmt_cards(board_cards)}", flush=True)
        else:
            print(f"{sep}{'─' * 44}", flush=True)

    def on_player_action(self, action: PlayerAction):
        seat   = "hero" if action.seat == 0 else f"seat {action.seat}"
        name   = f"({action.username})" if action.username else ""
        amount = f"  {self._fmt_amount(action.amount)}BB" if action.amount is not None else ""
        dollar = (f"  (${self._fmt_amount(action.amount_dollars)})"
                  if action.amount_dollars is not None else "")
        colour = "yellow" if action.seat == 0 else ""

        if self.verbose:
            parts = []
            if action.stack_before is not None and action.stack_after is not None:
                parts.append(
                    f"stack {self._fmt_amount(action.stack_before)}BB"
                    f" → {self._fmt_amount(action.stack_after)}BB"
                    f"  Δ{self._fmt_amount(action.stack_before - action.stack_after)}BB"
                )
            if action.street_total is not None:
                total_str = f"street total {self._fmt_amount(action.street_total)}BB"
                if action.street_total_dollars is not None:
                    total_str += f" (${self._fmt_amount(action.street_total_dollars)})"
                parts.append(total_str)
            stack_info = f"  [{' | '.join(parts)}]" if parts else ""
        else:
            stack_info = ""

        self._log("PLAYER ACTION",
                  f"{seat:<8}  {name:<22}  {action.action:<8}{amount}{dollar}{stack_info}",
                  colour)

    def on_your_turn(self, hand: HandState):
        recent = hand.all_actions[-5:] if hand.all_actions else []
        context = ", ".join(
            f"seat{a.seat} {a.action}" + (f" {self._fmt_amount(a.amount)}BB" if a.amount else "")
            for a in recent if a.seat != 0
        ) or "none"
        self._log("YOUR TURN", f"context=[{context}]", "green")

    def on_hand_complete(self, hand: HandState):
        board  = " ".join(hand.community_cards) if hand.community_cards else "—"
        bb_str = f"  BB={hand.bb_amount}" if hand.bb_amount else ""
        folded = "  (hero folded)" if hand.hero_folded else ""
        self._log("HAND COMPLETE",
                  f"{len(hand.all_actions)} actions  board=[{board}]{bb_str}{folded}",
                  "bold")
        print(f"  {'━'*66}\n", flush=True)

    def log(self, tag: str, msg: str, colour: str = ""):
        """Public helper for external callers (e.g. ScreenshotCapture)."""
        self._log(tag, msg, colour)

    @staticmethod
    def colour(text: str, name: str) -> str:
        """Wrap text in an ANSI colour by name (green/yellow/red/cyan/bold/dim)."""
        code = _COLOUR_CODES.get(name, "")
        return f"{code}{text}{_RESET}" if code else text

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ts() -> str:
        return time.strftime("%H:%M:%S")

    @staticmethod
    def _fmt_amount(v: float) -> str:
        return f"{v:g}"

    @staticmethod
    def _fmt_card(card_str: str) -> str:
        if len(card_str) < 2:
            return card_str
        colour = _SUIT_COLOUR.get(card_str[-1].lower(), "")
        return f"{_BOLD}{colour}{card_str}{_RESET}"

    @classmethod
    def _fmt_cards(cls, cards: list) -> str:
        return "  ".join(cls._fmt_card(str(c)) for c in cards)

    def _log(self, tag: str, msg: str, colour: str = ""):
        c = _COLOUR_CODES.get(colour, "")
        print(f"  {self._ts()}  {c}{tag:<16}{_RESET}  {msg}", flush=True)
