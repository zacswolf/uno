from card import Card
from enums import Color, Type
import random


class Deck:
    def __init__(self, with_replacement: bool = False) -> None:
        self.with_replacement = with_replacement

        self.deck: list[Card] = []

        for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]:
            # Only one ZERO per color
            self.deck.append(Card(Type.ZERO, color))

            # 2 per color
            for type in [
                Type.ONE,
                Type.TWO,
                Type.THREE,
                Type.FOUR,
                Type.FIVE,
                Type.SIX,
                Type.SEVEN,
                Type.EIGHT,
                Type.NINE,
                Type.REVERSE,
                Type.SKIP,
                Type.DRAW2,
            ]:
                self.deck.append(Card(type, color))
                self.deck.append(Card(type, color))

        # 4 of each wild card
        for type in [Type.CHANGECOLOR, Type.DRAW4]:
            for _ in range(4):
                self.deck.append(Card(type, Color.WILD))

        random.shuffle(self.deck)

    def draw_card(self) -> Card:
        """Draw a card

        Returns:
            Card: Drawn card
        """
        assert len(self.deck) > 0
        if not self.with_replacement:
            return self.deck.pop()
        else:
            return random.choice(self.deck)

    def size(self) -> int:
        """Get length of deck

        Returns:
            int: Number of cards in the deck
        """
        return len(self.deck)

    def reset(self, cards: list[Card]) -> None:
        """Reset deck with new cards

        Args:
            cards (list[Card]): New cards
        """
        assert self.size() == 0
        self.deck = cards


if __name__ == "__main__":
    deck = Deck()
    print(deck.deck)
