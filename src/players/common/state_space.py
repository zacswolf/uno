from abc import ABC, abstractmethod
from card import Card
import numpy as np

from enums import Color, Type


class StateSpace(ABC):
    def __init__(self, args) -> None:
        super().__init__()
        self.num_players = args.num_players

    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_state(
        self, hand: list[Card], top_of_pile: Card, card_counts: list[int]
    ) -> np.ndarray:
        raise NotImplementedError()


class SSRep1(StateSpace):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.NUM_TYPES = 15
        self.NUM_TYPES_NON_WILD = 13
        self.NUM_COLORS = 5
        self.NUM_COLORS_NON_WILD = 4

        self.ss_length = (
            self.NUM_COLORS_NON_WILD
            + self.NUM_TYPES
            + self.NUM_COLORS_NON_WILD * self.NUM_TYPES_NON_WILD
            + (self.NUM_TYPES - self.NUM_TYPES_NON_WILD)  # 2
            + self.num_players
        )
        assert self.ss_length == 73 + self.num_players

    def size(self) -> int:
        return self.ss_length

    def get_state(self, hand: list[Card], top_of_pile: Card, card_counts: list[int]):
        # ASSUME: 2 players only

        # top of pile [onehot color, onehot type]
        #       colors are in enum order no wild
        #       type is in enum order
        # hand: a count array for every color/type combo
        #       color enum pri, type enum second
        # card counts [cc0: me, cc1]
        #       in rotation order

        # length should be num_players + 4colors + 15types + 4*13 + 2
        assert len(card_counts) == self.num_players

        ss = np.zeros(self.ss_length)

        # top of pile
        assert top_of_pile.color != Color.WILD
        ss[top_of_pile.color] = 1
        ss[self.NUM_COLORS_NON_WILD + top_of_pile.type] = 1

        # Hand
        for card in hand:
            if card.color != Color.WILD:
                assert (card.type != Type.CHANGECOLOR) and (card.type != Type.DRAW4)
                ss[
                    self.NUM_COLORS_NON_WILD
                    + self.NUM_TYPES
                    + card.color * self.NUM_TYPES_NON_WILD
                    + card.type
                ] += 1
            else:
                assert (card.type == Type.CHANGECOLOR) or (card.type == Type.DRAW4)
                ss[
                    self.NUM_COLORS_NON_WILD
                    + self.NUM_TYPES
                    + self.NUM_COLORS_NON_WILD * self.NUM_TYPES_NON_WILD
                    + (card.type - Type.CHANGECOLOR)
                ] += 1

        # card counts
        ss[-len(card_counts) :] = card_counts

        return ss


# Inject game knowledge
# ss1 except with rotated order based off of top of pile and/or hand
#       color/ card numbers to current top of pile
#       random ordering if it doesnt matter
# represent hand as a graph
#   every card in your hand is a node
#   edges based off of if the card can be legally played on other cards in your hand
#   maybe something special for the top of pile
#   add meta data to each handcard/slot that says how many cards in your hand can be softstreaked with that card
#  Montecarlo end games


# How should the output of a bot's value-net look like?
#   A distribution over the whole card action space (doesn't matter if it has the card)
#   A distribution over the hand card action space
#   A distribution over the whole card action space except that the cards it has are at the front (rotated)

# The wild card out puts should be colored not wild
