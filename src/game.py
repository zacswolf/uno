from deck import Deck
from players.human import HumanPlayer
from enums import Direction, Color, Type
from card import Card
import random


class Game(object):
    def __init__(self, args) -> None:
        # Store args
        self.num_players = args.num_players
        self.skip_draw = not args.no_draw_skip

        if self.num_players < 2 or self.num_players > 10:
            raise Exception("Invalid number of players")

        self.deck = Deck(args.with_replacement)

        self.players = []
        for _ in range(self.num_players):
            self.players.append(HumanPlayer())

        # deal
        for _ in range(7):
            for player in self.players:
                player.get_card(self.draw_card())

        # draw top card
        self.pile = [self.draw_card()]
        while self.pile[-1].type > 9:
            self.pile.append(self.draw_card())

        self.direction = Direction.CLOCKWISE
        self.player_idx = 0

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
                next_player_idx = (self.player_idx + self.direction) % self.num_players

                card = self.turn()

                if card is not None:
                    # self.pile.append(self.draw_card())
                    self.pile.append(card)

                    skip = False

                    if card.type == Type.REVERSE:
                        # Reverse
                        self.direction = (
                            Direction.CLOCKWISE
                            if (self.direction == Direction.COUNTERCLOCKWISE)
                            else Direction.COUNTERCLOCKWISE
                        )
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
                        next_player_idx = (
                            next_player_idx + self.direction
                        ) % self.num_players

                if len(self.players[self.player_idx].hand) == 0:
                    game_over = True

                    # Notify players for feedback
                    for (idx, player) in enumerate(self.players):
                        player.on_finish((self.player_idx - idx) % self.num_players)
                else:
                    self.player_idx = next_player_idx
        except Exception:
            # Notify players for feedback
            for player in self.players:
                player.on_finish(-1)

            # Rethrow exception
            raise

        print("Game over")
        return

    def turn(self) -> Card:
        player = self.players[self.player_idx]
        card_counts = list(map(lambda player: len(player.hand), self.players))
        card_counts = card_counts[self.player_idx :] + card_counts[: self.player_idx]

        valid_card = False
        card = None
        while not valid_card:
            card = player.on_turn(self.pile, card_counts)
            if card is None:
                # Draw
                player.get_card(self.draw_card())
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
                else:
                    player.on_card_rejection(
                        card
                    )  # might have some flawed logic with re drawing
                    print("bad card")
            else:  # card is None:
                valid_card = True

        return card


if __name__ == "__main__":
    import argparse

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
        action="store_false",
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

    args = my_parser.parse_args()

    print("asd", args.with_replacement)
    assert args.num_players == 2

    game = Game(args)
    game.run_game()
