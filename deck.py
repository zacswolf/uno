from card import Card
from enums import Color, Type
import random



class Deck(object):
    def __init__(self) -> None:
        self.deck = []

        for color in [Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW]:
            # Only one ZERO per color
            self.deck.append(Card(Type.ZERO, color))

            # 2 per color
            for type in [Type.ONE, Type.TWO, Type.THREE, Type.FOUR, Type.FIVE, Type.SIX, 
                        Type.SEVEN, Type.EIGHT, Type.NINE, Type.REVERSE, Type.SKIP, Type.DRAW2]:
                self.deck.append(Card(type, color))
                self.deck.append(Card(type, color))
        
        # 4 of each wild card
        for type in [Type.CHANGECOLOR, Type.DRAW4]:
            for _ in range(4):
                self.deck.append(Card(type, Color.WILD))

        random.shuffle(self.deck)
    
    def draw_card(self) -> Card:
        if (len(self.deck) > 0):
            return self.deck.pop()
        raise Exception("No cards in deck")

if __name__ == "__main__":
    deck = Deck()
    print(deck.deck)