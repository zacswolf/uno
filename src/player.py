from enums import Color, Type
from card import Card
from typing import Callable
from abc import ABC, abstractmethod

from load_args import ArgsGameShared, ArgsPlayer


class Player(ABC):
    def __init__(self, player_args: ArgsPlayer, game_args: ArgsGameShared) -> None:
        self.hand: list[Card] = []

        self.player_args = player_args
        self.game_args = game_args

    def get_card(self, card: Card) -> None:
        """Called when Player gets a new Card

        Args:
            card (Card): Card that the player recieves
        """
        self.hand.append(card)

    @abstractmethod
    def on_turn(self, pile: list[Card], card_counts: list[int], drawn: bool) -> Card | None:
        """Called when its Player's turn

        Args:
            pile (list[Card]): The pile
            card_counts (list[int]): List of all players card counts relative to the player
            drawn (bool): True if the player just drew

        Raises:
            NotImplementedError: Inherited Player must impliment

        Returns:
            Card | None: Card to play card or None to draw
        """

        raise NotImplementedError()

    # Called when game rejects player's card
    def on_card_rejection(self, card: Card) -> None:
        """Called when game rejects player's card

        Args:
            card (Card): Card that the player tried to play
        """
        self.hand.append(card)

    # Winner is a player index relative to player or -1 if no-one wins
    def on_finish(self, winner: int) -> None:
        """Called when game is done or there is an error, optional override

        Args:
            winner (int): A player index relative to player or -1 if no-one wins
        """


def str_to_player(plyr_str: str) -> Callable[[], Player]:
    """Converts player string into a player 

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
        case "draw":
            from players.non_rl import DrawPlayer
            return DrawPlayer
        case "random":
            from players.non_rl import RandomPlayer
            return RandomPlayer
        case "noob":
            from players.non_rl import NoobPlayer
            return NoobPlayer
        case "basic":
            from players.non_rl import BasicPlayer
            return BasicPlayer
        case "decent":
            from players.non_rl import DecentPlayer
            return DecentPlayer
        case "decent2":
            from players.non_rl import DecentPlayer2
            return DecentPlayer2
        case "decent3":
            from players.non_rl import DecentPlayer3
            return DecentPlayer3
        case "decent4":
            from players.non_rl import DecentPlayer4
            return DecentPlayer4
        case "firstrlplayer":
            from players.first_rl_nn import FirstRLPlayer
            return FirstRLPlayer
        case "secrlplayer":
            from players.first_rl_nn import SecondRLPlayer
            return SecondRLPlayer
        case "reinvalact":
            from players.reinforce import ReinforceValActions
            return ReinforceValActions
        case "reinvalactsoft":
            from players.reinforce import ReinforceValActionsSoftmax
            return ReinforceValActionsSoftmax
        case "reinvalactsoft2":
            from players.reinforce import ReinforceValActionsSoftmax2
            return ReinforceValActionsSoftmax2
        case "onestepac":
            from players.actorcritic import OneStepActorCritic
            return OneStepActorCritic
        case "onestepacsoft":
            from players.actorcritic import OneStepActorCriticSoft
            return OneStepActorCriticSoft
        case "qlearn":
            from players.qlearning import QLearner
            return QLearner

    raise ValueError("player string `%s` is invalid" % plyr_str)
