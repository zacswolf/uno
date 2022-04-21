from argparse import Namespace
from deck import Deck
from enums import Direction, Color, Type
from card import Card
import random
import logging

from player import str_to_player, Player


class Game:
    def __init__(self, args: Namespace) -> None:
        try:
            logging.info(msg="args: %s" % args)

            # Store args
            self.num_players = args.num_players
            self.skip_draw = not args.no_draw_skip

            if self.num_players < 2 or self.num_players > 10:
                raise ValueError("Invalid number of players")
            elif self.num_players != len(args.players):
                raise ValueError("num_players arg doesn't match players arg list")

            self.players: list[Player] = []
            for player_str in args.players:
                # Create player based on player str
                self.players.append(str_to_player(player_str)(args))

            self.reset()
        except:
            logging.exception("Fatal error in initalizing game", exc_info=True)
            raise

    def reset(self):
        self.deck = Deck(args.with_replacement)

        # Clear hands
        for player in self.players:
            player.hand = []

        # Deal
        for _ in range(args.num_cards):
            for player in self.players:
                player.get_card(self.draw_card())
        for player_idx, player in enumerate(self.players):
            logging.info("%s: %s" % (player_idx, player.hand))

        # Draw top card
        self.pile = [self.draw_card()]
        while self.pile[-1].type > Type.NINE:
            self.pile.append(self.draw_card())

        self.direction = Direction.CLOCKWISE
        self.player_idx = 0  # Maybe make this random
        self.turn_num = 0

    def draw_card(self) -> Card:
        if not self.deck.size():
            # Deck is empty, move pile into deck

            self.deck.reset(self.pile[:-1])
            self.pile = [self.pile[-1]]

            if not self.deck.size():
                # Players are holding all of the cards
                raise Exception("Shitty players")
        card = self.deck.draw_card()
        assert bool(card.type >= Type.CHANGECOLOR) != bool(card.color != Color.WILD), (
            "Wild card has a color: %s" % card
        )
        return card

    def run_game(self) -> None:
        """Run the Game"""
        winner_idx = -1
        try:
            game_over = False
            while not game_over:
                self.turn_num += 1
                if self.turn_num % 10 == 0:
                    print("tn", self.turn_num)
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
                            "The next player at idx %s get's skipped" % next_player_idx
                        )
                        next_player_idx = (
                            next_player_idx + self.direction
                        ) % self.num_players

                if len(self.players[self.player_idx].hand) == 0:
                    # Current Player won the game
                    game_over = True
                    winner_idx = self.player_idx
                    logging.info(
                        "Game over! Winner deats: turn_num: %s, player_idx: %s, plyr_str: %s"
                        % (
                            self.turn_num,
                            self.player_idx,
                            self.players[self.player_idx].get_name(),
                        )
                    )
                    print(
                        "Game over! Winner deats: turn_num: %s, player_idx: %s, plyr_str: %s"
                        % (
                            self.turn_num,
                            self.player_idx,
                            self.players[self.player_idx].get_name(),
                        )
                    )

                    # Notify players for feedback
                    for (idx, player) in enumerate(self.players):
                        # TODO: Maybe rotate player idx to direction of play
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

        return winner_idx

    def turn(self) -> Card | None:
        """A turn of the game

        Returns:
            Card | None: card played on this turn or None if player passed
        """
        # Get current player
        player = self.players[self.player_idx]

        # Get all players num cards
        card_counts = list(map(lambda player: len(player.hand), self.players))
        logging.debug("Card Counts: %s" % card_counts)
        # Shift such that current player is first
        # TODO: Maybe rotate idx to direction of play
        card_counts = card_counts[self.player_idx :] + card_counts[: self.player_idx]

        logging.info(
            "turn_num: %s, player_idx: %s, plyr_str: %s"
            % (self.turn_num, self.player_idx, player.get_name())
        )

        valid_card = False
        card = None
        while not valid_card:
            assert card_counts[0] == len(player.hand)
            card = player.on_turn(self.pile, card_counts)
            # TODO: Make sure player isn't playing a card that they don't have
            if card is not None:
                assert card_counts[0] == len(player.hand) + 1

            if card is None:
                assert len(player.hand) == card_counts[0]
                # Draw
                card_drawn = self.draw_card()
                logging.debug("Draws %s" % card_drawn)
                player.get_card(card_drawn)
                assert len(player.hand) == card_counts[0] + 1
                card_counts[0] = len(player.hand)
                card = player.on_draw(self.pile, card_counts)
            # If card is none at this point, the player drew a card and didn't play it

            # Check card
            if card is not None:
                assert card_counts[0] == len(player.hand) + 1

                if card.can_play_on(self.pile[-1]):
                    valid_card = True

                    # Deal with wild
                    if card.color == Color.WILD:

                        color = player.on_choose_wild_color(
                            self.pile, card_counts, card.type
                        )
                        logging.debug("A wild was played: %s" % color)
                        # TODO: Make sure Color isn't Wild
                        card.color = color

                    logging.debug(
                        "Card played: %s, Top of pile: %s" % (card, self.pile[-1])
                    )
                else:
                    logging.warning(
                        "Card rejected: %s, Top of pile: %s" % (card, self.pile[-1])
                    )
                    player.on_card_rejection(
                        card
                    )  # might have some flawed logic with re drawing
                    assert card_counts[0] == len(self.players[self.player_idx].hand)
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
    my_parser.add_argument(
        "--num_games",
        type=int,
        default=1,
        help="Number of games to play",
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
    winner_tracker = [0] * (args.num_players + 1)
    for game_num in range(args.num_games):
        logging.info("STARTING GAME %d" % game_num)
        winner_idx = game.run_game()
        winner_tracker[winner_idx] += 1
        game.reset()
    print(winner_tracker)
    logging.info("Winner tracker: %s" % winner_tracker)
