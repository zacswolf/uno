from enums import Color
from card import Card


class Player(object):
    def __init__(self) -> None:
        self.hand = []
    
    # Return a string for logging that identifies the player, use plyr_str
    def get_name(self) -> str:
        raise NotImplementedError()

    # Called when Player gets a new Card
    def get_card(self, card) -> None:
        self.hand.append(card)

    # Called when its Player's turn
    # Return Card to play card or None to draw
    def on_turn(self, pile, card_counts) -> Card:
        raise NotImplementedError()

    # Called after Player draws a card and needs to make a decision
    # Return Card to play card or None to skip
    def on_draw(self, pile, card_counts) -> None:
        raise NotImplementedError()

    # Called after Player plays a wild card
    def on_choose_wild_color(self, pile, card_counts, type) -> Color:
        raise NotImplementedError()

    # Called when game rejects player's card
    def on_card_rejection(self, card) -> None:  # self, pile, card_counts, card
        self.hand.append(card)

    # Called when game is done or there is an error
    # Winner is a player index relative to player or -1 if no-one wins
    def on_finish(self, winner) -> None:
        return None


# To add a new bot add a case to the and then return the Player
def str_to_player(plyr_str: str) -> Player:
    match plyr_str:
        case "human":
            from players.human import HumanPlayer 
            return HumanPlayer
    raise Exception("player string `%s` is invalid" % plyr_str)
