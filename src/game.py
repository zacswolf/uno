from argparse import Namespace
from deck import Deck
from enums import Direction, Color, Type
from card import Card
import random
import logging
from load_args import load_args

from player import str_to_player, Player


class Game:
    def __init__(self, args) -> None:
        try:
            logging.info(f"args: {args}")

            # Store args
            self.num_players = args.game.shared.num_players
            self.draw_skip = args.game.private.draw_skip
            self.alternate = args.game.private.alternate

            if self.num_players < 2 or self.num_players > 10:
                raise ValueError("Invalid number of players")
            elif self.num_players != len(args.players):
                raise ValueError(
                    f"num_players arg ({self.num_players}) doesn't match players arg list ({args.players})"
                )

            self.players: list[Player] = []
            self.player_names = []
            for player_idx, player_args in enumerate(args.players):
                # Create player based on player str
                player_args.player_idx = player_idx
                self.players.append(
                    str_to_player(player_args.player)(player_args, args.game.shared)
                )
                self.player_names.append(player_args.player)

            # self.reset(0)
        except:
            logging.exception("Fatal error in initalizing game", exc_info=True)
            raise

    def reset(self, game_num: int):
        self.deck = Deck(args.game.private.with_replacement)

        # Clear hands
        for player in self.players:
            player.hand = []

        # Deal
        for _ in range(args.game.private.num_cards):
            for player in self.players:
                player.get_card(self.draw_card())
        for player_idx, player in enumerate(self.players):
            logging.info(f"{player_idx}: {player.hand}")

        # Draw top card
        self.pile = [self.draw_card()]
        while self.pile[-1].type > Type.NINE:
            self.pile.append(self.draw_card())

        logging.info(f"Top Card: {self.pile[-1]}")

        self.direction = Direction.CLOCKWISE
        if self.alternate:
            self.player_idx = game_num % len(self.players)
        else:
            self.player_idx = 0
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
        assert bool(card.type >= Type.CHANGECOLOR) != bool(
            card.color != Color.WILD
        ), f"Wild card has a color: {card}"
        return card

    def run_game(self) -> None:
        """Run the Game"""
        winner_idx = -1
        num_turns = -1
        winner_str = ""
        try:
            game_over = False
            while not game_over:
                self.turn_num += 1
                if self.turn_num % 100 == 0:
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

                        # Skip if self.draw_skip
                        skip = self.draw_skip

                    elif card.type == Type.DRAW4:
                        # Draw 4
                        for _ in range(4):
                            self.players[next_player_idx].get_card(self.draw_card())

                        # Skip if self.draw_skip
                        skip = self.draw_skip

                    if skip:
                        # Skip logic
                        logging.debug(f"Skipped {next_player_idx}")
                        next_player_idx = (
                            next_player_idx + self.direction
                        ) % self.num_players

                if len(self.players[self.player_idx].hand) == 0:
                    # Current Player won the game
                    game_over = True
                    winner_idx = self.player_idx
                    num_turns = self.turn_num
                    winner_str = self.player_names[self.player_idx]

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

        return winner_idx, num_turns, winner_str

    def turn(self) -> Card | None:
        """A turn of the game

        Returns:
            Card | None: card played on this turn or None if player passed
        """
        # Get current player
        player = self.players[self.player_idx]
        player_name = self.player_names[self.player_idx]

        # Get all players num cards
        card_counts = list(map(lambda player: len(player.hand), self.players))
        logging.debug("Card Counts: %s" % card_counts)
        # Shift such that current player is first
        # TODO: Maybe rotate idx to direction of play
        card_counts = card_counts[self.player_idx :] + card_counts[: self.player_idx]

        logging.info(
            "turn_num: %s, player_idx: %s, plyr_str: %s"
            % (self.turn_num, self.player_idx, player_name)
        )

        valid_card = False
        card = None
        while not valid_card:
            assert card_counts[0] == len(player.hand)
            card = player.on_turn(self.pile, card_counts, False)
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
                card = player.on_turn(self.pile, card_counts, True)
            # If card is none at this point, the player drew a card and didn't play it

            # Check card
            if card is not None:
                assert card_counts[0] == len(player.hand) + 1

                if card.can_play_on(self.pile[-1]):
                    valid_card = True

                    assert card.color != Color.WILD, "Wild wasn't colored"

                    if card.type >= Type.CHANGECOLOR:
                        # Wild is already colored
                        logging.debug("A wild was played: %s" % card.color)

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
    import sys
    import os

    if not sys.version_info >= (3, 10):
        sys.exit("Python < 3.10 is unsupported.")

    args = load_args()
    # Setup logging
    log_file = os.path.join(
        args.game.private.root_file, "../logs/", f"log_{args.game.shared.run_name}.log"
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
    winner_tracker = [0] * (args.game.shared.num_players + 1)
    for game_num in range(args.game.shared.num_games):
        logging.info("STARTING GAME %d" % game_num)
        game.reset(game_num)
        winner_idx, num_turns, winner_str = game.run_game()
        winner_tracker[winner_idx] += 1
        game_res_str = f"Game over! {game_num}:{winner_tracker} Winner deats: player_idx: {winner_idx}, plyr_str: {winner_str}, turn_num: {num_turns}"
        logging.info(game_res_str)
        print(game_res_str)
    print(winner_tracker)
    logging.info(f"Winner tracker: {winner_tracker}")
