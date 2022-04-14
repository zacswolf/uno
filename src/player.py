from enums import Color
from card import Card
from typing import Callable


class Player(object):
    def __init__(self) -> None:
        self.hand: list[Card] = []
    
    def get_name(self) -> str:
        """Identifies the player for logging

        Raises:
            NotImplementedError: Inherited Player must impliment

        Returns:
            str: Name of Player, use plyr_str
        """
        raise NotImplementedError()

    def get_card(self, card: Card) -> None:
        """Called when Player gets a new Card

        Args:
            card (Card): Card that the player recieves
        """
        self.hand.append(card)

    def on_turn(self, pile: list[Card], card_counts: list[int]) -> Card | None:
        """Called when its Player's turn

        Args:
            pile (list[Card]): The pile
            card_counts (list[int]): List of all players card counts relative to the player

        Raises:
            NotImplementedError: Inherited Player must impliment

        Returns:
            Card | None: Card to play card or None to draw
        """
        raise NotImplementedError()

    def on_draw(self, pile: Card, card_counts: list[int]) -> Card | None:
        """Called after Player draws a card and needs to make a decision

        Args:
            pile (Card): Card on the top of the pile
            card_counts (list[int]): List of all players card counts relative to the player

        Raises:
            NotImplementedError: Inherited Player must impliment

        Returns:
            Card | None: Card to play card or None to pass
        """
        raise NotImplementedError()

    # Called after Player plays a wild card
    def on_choose_wild_color(self, pile: Card, card_counts: list[int]) -> Color:
        """Called after Player draws a card and needs to make a decision

        Args:
            pile (Card): Card on the top of the pile
            card_counts (list[int]): List of all players card counts relative to the player

        Raises:
            NotImplementedError: Inherited Player must impliment

        Returns:
            Color: Desired color of the wildcard
        """
        raise NotImplementedError()

    # Called when game rejects player's card
    def on_card_rejection(self, card: Card) -> None:  # self, pile, card_counts, card
        """Called when game rejects player's card

        Args:
            card (Card): Card that the player tried to play
        """
        self.hand.append(card)

    # Winner is a player index relative to player or -1 if no-one wins
    def on_finish(self, winner:int) -> None:
        """Called when game is done or there is an error, optional override

        Args:
            winner (int): A player index relative to player or -1 if no-one wins
        """


def str_to_player(plyr_str: str) -> Callable[[], Player]:
    """Converts player string into 

    Args:
        plyr_str (str): String representing a type of player

    Raises:
        ValueError: Player string is invalid

    Returns:
        Callable[[], Player]: Object type of a inherited Player class
    """
    
    # To add a new bot add a case to the and then return the Player
    match plyr_str:
        case "human":
            from players.human import HumanPlayer 
            return HumanPlayer
    raise ValueError("player string `%s` is invalid" % plyr_str)
