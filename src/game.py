from deck import Deck
from enums import Direction, Color, Type
from card import Card
import random
import logging
from typing import List

from player import str_to_player, Player


class Game(object):
    def __init__(self, args) -> None:
        try:
            logging.info(msg="args: %s" % args)

            # Store args
            self.num_players = args.num_players
            self.skip_draw = not args.no_draw_skip

            if self.num_players < 2 or self.num_players > 10:
                raise ValueError("Invalid number of players")
            elif self.num_players != len(args.players):
                raise ValueError("num_players arg doesn't match players arg list")

            self.deck = Deck(args.with_replacement)

            self.players: List[Player] = []
            for player in args.players:
                # Create player based on player str
                self.players.append(str_to_player(player)())

            # Deal
            for _ in range(args.num_cards):
                for player in self.players:
                    player.get_card(self.draw_card())

            # Draw top card
            self.pile = [self.draw_card()]
            while self.pile[-1].type > Type.NINE:
                self.pile.append(self.draw_card())

            self.direction = Direction.CLOCKWISE
            self.player_idx = 0  # Maybe make this random
            self.turn_num = 0
        except:
            logging.exception("Fatal error in initalizing game", exc_info=True)
            raise

    def draw_card(self) -> Card:
        if not self.deck.size():
            # Deck is empty, move pile into deck

            self.deck = random.shuffle(self.pile[:-1])
            self.pile = [self.pile[-1]]

            if not self.deck.size():
                # players are holding all of the cards
                raise Exception("Shitty players")

        return self.deck.draw_card()

    def run_game(self):
        game_over = False
        try:
            while not game_over:
                self.turn_num += 1
                card = self.turn()

                next_player_idx = (self.player_idx + self.direction) % self.num_players

                if card is not None:
                    self.pile.append(card)

                    skip = False

                    if card.type == Type.REVERSE:
                        # Reverse
                        self.direction = (
                            Direction.CLOCKWISE
                            if (self.direction == Direction.COUNTERCLOCKWISE)
                            else Direction.COUNTERCLOCKWISE
                        )
                        # Change next player
                        next_player_idx = (
                            self.player_idx + self.direction
                        ) % self.num_players
                        if self.num_players == 2:
                            # Reverse 1v1 acts like a skip
                            skip = True
                    elif card.type == Type.SKIP:
                        # Skip
                        skip = True
                    elif card.type == Type.DRAW2:
                        # Draw 2
                        for _ in range(2):
                            self.players[next_player_idx].get_card(self.draw_card())

                        # Skip if self.skip_draw
                        skip = self.skip_draw

                    elif card.type == Type.DRAW4:
                        # Draw 4
                        for _ in range(4):
                            self.players[next_player_idx].get_card(self.draw_card())

                        # Skip if self.skip_draw
                        skip = self.skip_draw

                    if skip:
                        # Skip logic
                        logging.debug(
                            "The next player at idx %s can't act" % next_player_idx
                        )
                        next_player_idx = (
                            next_player_idx + self.direction
                        ) % self.num_players

                if len(self.players[self.player_idx].hand) == 0:
                    # Current Player won the game
                    game_over = True
                    logging.info(
                        "Game over! Winner deats: turn_num: %s, player_idx: %s, plyr_str: %s"
                        % (
                            self.turn_num,
                            self.player_idx,
                            self.players[self.player_idx].get_name(),
                        )
                    )

                    # Notify players for feedback
                    for (idx, player) in enumerate(self.players):
                        player.on_finish((self.player_idx - idx) % self.num_players)
                else:
                    # Next player's turn
                    self.player_idx = next_player_idx
        except Exception:
            # Log
            logging.exception("Fatal error in run_game", exc_info=True)

            # Notify players for feedback
            for player in self.players:
                player.on_finish(-1)

            # Rethrow exception
            raise

        print("Game over")
        return

    def turn(self) -> Card:
        # Get current player
        player = self.players[self.player_idx]

        # Get all players num cards
        card_counts = list(map(lambda player: len(player.hand), self.players))
        # Shift such that current player is first
        # TODO: rotate by direction of play
        card_counts = card_counts[self.player_idx :] + card_counts[: self.player_idx]

        logging.info(
            "turn_num: %s, player_idx: %s, plyr_str: %s"
            % (self.turn_num, self.player_idx, player.get_name())
        )

        valid_card = False
        card = None
        while not valid_card:
            card = player.on_turn(self.pile, card_counts)
            # TODO: make sure player isn't playing a card that they don't have

            if card is None:
                # Draw
                card_drawn = self.draw_card()
                logging.debug("Draws %s" % card_drawn)
                player.get_card(card_drawn)
                assert card_counts[0] == len(self.players[self.player_idx].hand) - 1
                card_counts[0] = len(self.players[self.player_idx].hand)
                card = player.on_draw(self.pile, card_counts)
            # If card is none at this point, the player drew a card and didn't play it

            # Check card
            if card is not None:
                if card.canPlayOn(self.pile[-1]):
                    valid_card = True

                    # Deal with wild
                    if card.color == Color.WILD:
                        color = player.on_choose_wild_color(
                            self.pile, card_counts, card.type
                        )
                        card.color = color

                    logging.debug(
                        "Card played: %s, Top of pile: %s" % (card, self.pile[-1])
                    )

                else:
                    logging.warn(
                        "Card rejected: %s, Top of pile: %s" % (card, self.pile[-1])
                    )
                    player.on_card_rejection(
                        card
                    )  # might have some flawed logic with re drawing
            else:
                # card is None thus the player just drawed
                valid_card = True

        return card


if __name__ == "__main__":
    import argparse
    import sys
    import os
    from datetime import datetime

    if not sys.version_info >= (3, 10):
        sys.exit("Python < 3.10 is unsupported.")

    my_parser = argparse.ArgumentParser(description="Uno game")
    my_parser.add_argument(
        "-n",
        "--num_players",
        type=int,
        default=2,
        choices=range(2, 11),
        metavar="[2-10]",
        help="Number of players",
    )
    my_parser.add_argument(
        "--no_draw_skip",
        action="store_true",
        help="Don't skip a players turn if they have to draw 2/4",
    )
    my_parser.add_argument(
        "--with_replacement",
        action="store_true",
        help="Deck is drawn with replacement",
    )

    my_parser.add_argument(
        "--num_cards",
        type=int,
        default=7,
        help="Initial number of cards per player",
    )
    my_parser.add_argument(
        "--players",
        nargs="+",
        required=True,
        help="List of players using player strings",
    )

    args = my_parser.parse_args()

    # assert args.num_players == 2

    # Setup logging
    log_file = os.path.join(
        os.path.dirname(__file__),
        "../logs/",
        datetime.now().strftime("log_%m_%d_%H_%M_%S.log"),
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=log_file,
        filemode="w",
    )

    # Main
    game = Game(args)
    game.run_game()
