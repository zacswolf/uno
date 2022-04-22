from enums import Color, Type


class Card:
    def __init__(self, type: Type, color: Color) -> None:
        self.type = type
        self.color = color

    def can_play_on(self, card: "Card") -> bool:
        """Checks if this card can be played on another card

        Args:
            card (Card): Another card

        Returns:
            bool: Can this card be played on another card
        """
        if self.color == card.color or self.type == card.type:
            return True
        if self.color == Color.WILD or card.color == Color.WILD:
            return True

        if self.type >= Type.CHANGECOLOR:
            return True

        return False

    def __str__(self) -> str:
        return "(%s, %s)" % (self.type.name, self.color.name)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Card):
            if self.type >= Type.CHANGECOLOR:
                # Wild card so we ignore the color
                return self.type == other.type

            return self.type == other.type and self.color == other.color
        return False


if __name__ == "__main__":
    card = Card(Type.ZERO, Color.RED)
    print(card)
