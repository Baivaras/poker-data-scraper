"""Hand state dataclasses — Phase 3.

Tracks everything from HAND_START to HAND_COMPLETE:
  - Hero cards + table state (from Phase 1 & 2)
  - All player actions across all streets
  - Hero-centric action rounds
  - Street progression
"""
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from core.models import Card
from core.table_state import TableState

ActionType = Literal["FOLD", "CHECK", "CALL", "RAISE", "BET", "ALL_IN"]
Street     = Literal["PREFLOP", "FLOP", "TURN", "RIVER"]


@dataclass
class PlayerAction:
    seat:           int             # 0 = hero, 1-5 = opponent seat
    username:       Optional[str]
    action:         ActionType
    amount:         Optional[float] # chips added this action in BBs (None for CHECK/FOLD)
    amount_bb:      Optional[float] # same as amount (BBs) — filled in Phase 4
    street:         Street
    timestamp:      int             # unix epoch
    amount_dollars:    Optional[float] = None  # amount × bb_dollar_value (filled in Phase 4)
    street_total:      Optional[float] = None  # total committed by this seat this street (BBs)
    street_total_bb:   Optional[float] = None  # same as street_total (filled in Phase 4)
    street_total_dollars: Optional[float] = None  # street_total × bb_dollar_value (Phase 4)
    stack_before:   Optional[float] = None  # stack before this action (BBs)
    stack_after:    Optional[float] = None  # stack after this action (BBs)

    def to_dict(self) -> dict:
        return {
            "seat":                  self.seat,
            "username":              self.username,
            "action":                self.action,
            "amount":                self.amount,
            "amount_bb":             self.amount_bb,
            "amount_dollars":        self.amount_dollars,
            "street_total":          self.street_total,
            "street_total_bb":       self.street_total_bb,
            "street_total_dollars":  self.street_total_dollars,
            "stack_before":          self.stack_before,
            "stack_after":           self.stack_after,
            "street":                self.street,
            "timestamp":             self.timestamp,
        }


@dataclass
class ActionRound:
    """One cycle from hero's last action to hero's next action."""
    street:              Street
    round_number:        int
    actions_before_hero: List[PlayerAction] = field(default_factory=list)
    hero_action:         Optional[PlayerAction] = None
    actions_after_hero:  List[PlayerAction] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "street":              self.street,
            "round_number":        self.round_number,
            "actions_before_hero": [a.to_dict() for a in self.actions_before_hero],
            "hero_action":         self.hero_action.to_dict() if self.hero_action else None,
            "actions_after_hero":  [a.to_dict() for a in self.actions_after_hero],
        }


@dataclass
class HandState:
    hand_id:         str
    hero_cards:      List[Card]
    table_state:     Optional[TableState]  # Phase 2 snapshot
    current_street:  Street = "PREFLOP"
    community_cards: List[str] = field(default_factory=list)  # Phase 5

    # Hero-centric grouped rounds
    action_rounds:   List[ActionRound] = field(default_factory=list)

    # Flat ordered list of every action — all seats, all streets
    # Includes actions after hero folded so we capture full hand history
    all_actions:     List[PlayerAction] = field(default_factory=list)

    bb_amount:       Optional[float] = None   # detected at hand start (Phase 4)
    pot_size:        Optional[float] = None
    hero_folded:     bool = False
    started_at:      int  = field(default_factory=lambda: int(time.time()))
    completed_at:    Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "hand_id":         self.hand_id,
            "hero_cards":      [c.to_dict() for c in self.hero_cards],
            "table_state":     self.table_state.to_dict() if self.table_state else None,
            "current_street":  self.current_street,
            "community_cards": self.community_cards,
            "action_rounds":   [r.to_dict() for r in self.action_rounds],
            "all_actions":     [a.to_dict() for a in self.all_actions],
            "bb_amount":       self.bb_amount,
            "pot_size":        self.pot_size,
            "hero_folded":     self.hero_folded,
            "started_at":      self.started_at,
            "completed_at":    self.completed_at,
        }
