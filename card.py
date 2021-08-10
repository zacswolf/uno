from enums import Color, Type

class Card(object):
    def __init__(self, type, color) -> None:
        self.type = type
        self.color = color
    
    def canPlayOn(self, top_of_pile) -> bool:
        if (self.color == top_of_pile.color or self.type == top_of_pile.type):
            return True
        if (self.color == Color.WILD):
            return True
        return False

    def __str__(self) -> str:
        return "(%s, %s)" % (self.type.name, self.color.name)

    def __repr__(self) -> str:
        return self.__str__()

if __name__ == "__main__":
    card = Card(Type.ZERO, Color.RED)
    print(card)