from card import Card
from enums import Color, Type
import numpy as np


def val_action_mask(
    hand: list[Card], top_of_pile: Card, action_space  #: ActionSpace
) -> np.ndarray:
    """Function to get a mask for the valid action space

    Args:
        hand (list[Card]): Player hand
        top_of_pile (Card): Card on the top of the pile
        action_space (ActionSpace): The action space

    Returns:
         np.ndarray: boolean action mask
    """
    as_size = action_space.size()

    cards = (
        action_space.idx_to_card(action_idx, hand, top_of_pile)
        for action_idx in range(as_size)
    )

    return np.array(
        [
            True if card is None else (card.can_play_on(top_of_pile) and card in hand)
            for card in cards
        ],
        dtype=bool,
    )


def color_map(card_color, top_color):
    assert top_color != Color.WILD
    return (card_color - top_color) % 4


def reverse_color_map(card_color, top_color):
    assert top_color != Color.WILD
    return (card_color + top_color) % 4
