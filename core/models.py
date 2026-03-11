"""Core data models for the poker tracker."""
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Card:
    rank: str  # 'A','2'..'9','T','J','Q','K'
    suit: str  # 'c','d','h','s'

    def to_str(self) -> str:
        return f"{self.rank}{self.suit}"

    def to_dict(self) -> dict:
        return {"rank": self.rank, "suit": self.suit}


@dataclass
class HeroCardsResult:
    cards: List[Card]
    confidence: float   # min(card1_conf, card2_conf), range 0–1
    timestamp: int      # unix epoch seconds
    hand_id: str

    def to_dict(self) -> dict:
        return {
            "hero_cards": [c.to_dict() for c in self.cards],
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "hand_id": self.hand_id,
        }
