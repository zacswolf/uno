"""
1. Define some state representation(s)

2. Most basic rl bot, what algos for this
    a) How to connect with player interface

3. How to do train infra

4. Making more rlbots, what algos to try

5...

"""


from card import Card
import numpy as np

from enums import Color, Type


def get_ss_rep1(hand, top_of_pile: Card, card_counts: list(int)):
    # ASSUME: 2 players only

    # card counts [cc0: me, cc1]
    #       in rotation order
    # top of pile [onehot color, onehot type]
    #       colors are in enum order no wild
    #       type is in enum order
    # hand: a count array for every color/type combo
    #       color enum pri, type enum second

    # length should be num_players + 4colors + 15types + 4*13 + 2

    NUM_TYPES = 15
    NUM_TYPES_NON_WILD = 13
    NUM_COLORS = 5
    NUM_COLORS_NON_WILD = 4

    ss_length = (
        len(card_counts)
        + NUM_COLORS_NON_WILD
        + NUM_TYPES
        + NUM_COLORS_NON_WILD * NUM_TYPES_NON_WILD
        + (NUM_TYPES - NUM_TYPES_NON_WILD)
    )
    assert ss_length == 73 + len(card_counts)

    ss = np.zeros(ss_length)

    # card counts
    ss[0 : len(card_counts)] = card_counts

    # top of pile
    ss[len(card_counts) + top_of_pile.color] = 1
    ss[len(card_counts) + NUM_COLORS_NON_WILD + top_of_pile.type] = 1

    # Hand
    for card in hand:
        if card.color != Color.Wild:
            ss[
                len(card_counts)
                + NUM_COLORS_NON_WILD
                + NUM_TYPES
                + card.color * NUM_TYPES_NON_WILD
                + card.type
            ] += 1
        else:
            assert (card.type == Type.CHANGECOLOR) or (card.type == Type.DRAW4)
            ss[card.Type - (Type.DRAW4 + 1)] += 1


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
