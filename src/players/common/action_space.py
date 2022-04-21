from abc import ABC, abstractmethod
from card import Card
import numpy as np

from enums import Color, Type


class ActionSpace(ABC):
    def __init__(self, args) -> None:
        super().__init__()

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def card_to_idx(self, card: Card | None) -> int:
        raise NotImplementedError()

    @abstractmethod
    def idx_to_card(self, idx: int) -> Card | None:
        raise NotImplementedError()


class ASRep1(ActionSpace):
    def __init__(self, args) -> None:
        super().__init__(args)
        # self.A_SIZE = 15 * 4 + 1  # plus one for the draw/noop

        self.NUM_TYPES = 15
        self.NUM_TYPES_NON_WILD = 13
        self.NUM_COLORS = 5
        self.NUM_COLORS_NON_WILD = 4

        self.as_length = self.NUM_COLORS_NON_WILD * self.NUM_TYPES + 1  # Draw/Noop

    def size(self) -> int:
        return self.as_length

    def card_to_idx(self, card: Card | None) -> int:
        if card:
            assert card.color != Color.WILD
            return card.color * self.NUM_TYPES + card.type
        else:
            return self.as_length - 1  # Drawing/Noop

    def idx_to_card(self, idx: int) -> Card | None:
        assert idx < self.as_length
        if idx != self.as_length - 1:
            color = Color(idx // self.NUM_TYPES)
            c_type = Type(idx % self.NUM_TYPES)
            return Card(c_type, color)
        return None
