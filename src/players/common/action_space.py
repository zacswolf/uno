from abc import ABC, abstractmethod
from card import Card
import numpy as np

from enums import Color, Type
from players.common.misc import color_map, reverse_color_map


class ActionSpace(ABC):
    def __init__(self, args) -> None:
        super().__init__()

        self.NUM_TYPES = Type.DRAW4 + 1
        self.NUM_TYPES_NON_WILD = Type.CHANGECOLOR
        self.NUM_TYPES_WILD = Type.DRAW4 - Type.CHANGECOLOR + 1
        self.NUM_COLORS = Color.WILD + 1
        self.NUM_COLORS_NON_WILD = Color.WILD

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def card_to_idx(
        self, card: Card | None, hand: list[Card] = None, top_of_pile: Card = None
    ) -> int:
        raise NotImplementedError()

    @abstractmethod
    def idx_to_card(
        self, idx: int, hand: list[Card] = None, top_of_pile: Card = None
    ) -> Card | None:
        raise NotImplementedError()


class ASRep1(ActionSpace):
    """We consider this to be a full representation of the state space"""

    def __init__(self, args) -> None:
        super().__init__(args)

        self.as_length = self.NUM_COLORS_NON_WILD * self.NUM_TYPES + 1  # Draw/Noop

    def size(self) -> int:
        return self.as_length

    def card_to_idx(
        self, card: Card | None, hand: list[Card] = None, top_of_pile: Card = None
    ) -> int:
        if card:
            assert card.color != Color.WILD
            return card.color * self.NUM_TYPES + card.type
        else:
            return self.as_length - 1  # Drawing/Noop

    def idx_to_card(
        self, idx: int, hand: list[Card] = None, top_of_pile: Card = None
    ) -> Card | None:
        assert idx < self.as_length
        if idx != self.as_length - 1:
            color = Color(idx // self.NUM_TYPES)
            c_type = Type(idx % self.NUM_TYPES)
            return Card(c_type, color)
        return None


class ASRep2(ActionSpace):
    """We consider this to be a full representation of the state space with color rotations
    ASRep1 with top_of_pile based misc.color_map
    """

    def __init__(self, args) -> None:
        super().__init__(args)

        self.as_length = self.NUM_COLORS_NON_WILD * self.NUM_TYPES + 1  # Draw/Noop

    def size(self) -> int:
        return self.as_length

    def card_to_idx(
        self, card: Card | None, hand: list[Card], top_of_pile: Card
    ) -> int:
        if card:
            assert card.color != Color.WILD
            return color_map(card.color, top_of_pile.color) * self.NUM_TYPES + card.type
        else:
            return self.as_length - 1  # Drawing/Noop

    def idx_to_card(self, idx: int, hand: list[Card], top_of_pile: Card) -> Card | None:
        assert idx < self.as_length
        if idx != self.as_length - 1:
            card_color = Color(idx // self.NUM_TYPES)
            color = Color(reverse_color_map(card_color, top_of_pile.color))
            c_type = Type(idx % self.NUM_TYPES)
            return Card(c_type, color)
        return None


class ASRep3WIP(ActionSpace):
    """Action space that doesn't include colors that are diff than top card for non-wild
    WIP because it doesn't support playing like a green 7 on a red 7
    """

    def __init__(self, args) -> None:
        super().__init__(args)

        self.as_length = (
            self.NUM_TYPES_NON_WILD + self.NUM_COLORS_NON_WILD * self.NUM_TYPES_WILD + 1
        )

    def size(self) -> int:
        return self.as_length

    def card_to_idx(
        self, card: Card | None, hand: list[Card], top_of_pile: Card
    ) -> int:
        if card:
            assert card.color != Color.WILD
            if card.type >= self.NUM_TYPES_NON_WILD:
                # Colored Wild
                return (
                    self.NUM_TYPES_NON_WILD
                    + card.color * self.NUM_TYPES_WILD
                    + (card.type - self.NUM_TYPES_NON_WILD)
                )
            else:
                assert (
                    card.color == top_of_pile.color
                ), "This action space assume same color as top_of_pile"
                return card.type
        else:
            return self.as_length - 1  # Drawing/Noop

    def idx_to_card(self, idx: int, hand: list[Card], top_of_pile: Card) -> Card | None:
        assert idx < self.as_length
        if idx < self.NUM_TYPES_NON_WILD:
            color = top_of_pile.color
            c_type = Type(idx)
            return Card(c_type, color)
        elif idx < self.as_length - 1:
            # Wild
            color = Color((idx - self.NUM_TYPES_NON_WILD) // self.NUM_TYPES_WILD)
            c_type = Type(
                ((idx - self.NUM_TYPES_NON_WILD) % self.NUM_TYPES_WILD)
                + self.NUM_TYPES_NON_WILD
            )
            return Card(c_type, color)
        return None
