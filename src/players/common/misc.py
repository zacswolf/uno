from card import Card
from enums import Color, Type


def act_filter(hand, card: Card | None, top_of_pile: Card):
    # True if can is in hand and can be played on top_of_pile
    if card is None:
        return True
    else:
        return card.can_play_on(top_of_pile) and card in hand


def color_map(card_color, top_color):
    assert top_color != Color.WILD
    return (card_color - top_color) % 4


def reverse_color_map(card_color, top_color):
    assert top_color != Color.WILD
    return (card_color + top_color) % 4
