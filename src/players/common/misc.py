from card import Card
from enums import Color, Type


def act_filter(hand, card: Card | None, top_of_pile: Card):
    # True if can is in hand and can be played on top_of_pile
    if card is None:
        return True
    else:
        if card.type >= Type.CHANGECOLOR:
            card = Card(card.type, Color.WILD)
        return card.can_play_on(top_of_pile) and card in hand
