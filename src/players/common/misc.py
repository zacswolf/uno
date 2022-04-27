from card import Card
from enums import Color, Type
import numpy as np


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

def sample(action_idxs, action_dist, epsilon=1):
    action_idx = -1
    if np.random.random() > epsilon:
        action_idx = action_idxs[np.argmax(action_dist)]
    else:
        action_idx = np.random.choice(action_idxs, p=action_dist)
    return action_idx