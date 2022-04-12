from enums import Color
from card import Card


class Player(object):
    def __init__(self) -> None:
        self.hand = []

    def get_card(self, card) -> None:
        self.hand.append(card)

    # Return Card to play card or None to draw
    def on_turn(self, pile, card_counts) -> Card:
        raise NotImplementedError()

    # Return Card to play card or None to skip
    def on_draw(self, pile, card_counts) -> None:
        raise NotImplementedError()

    def on_choose_wild_color(self, pile, card_counts, type) -> Color:
        raise NotImplementedError()

    # Game rejects player's card
    def on_card_rejection(self, card) -> None:  # self, pile, card_counts, card
        self.hand.append(card)
        print("Refunded")

    # Winner is a player index relative to player or -1 if no-one wins
    def on_finish(self, winner) -> None:
        return None

def str_to_player(plyr_str: str) -> Player:
    """
    To add a new bot add a case to the and then return the Player
    """

    match plyr_str:
        case "human":
            from players.human import HumanPlayer 
            return HumanPlayer
    raise Exception("player string `%s` is invalid" % plyr_str)
