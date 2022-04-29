from collections import Counter
from copy import copy
import logging

from card import Card
from player import Player
from enums import Color, Type
import random


class DrawPlayer(Player):
    """Player that always draws"""

    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        return None

    def on_card_rejection(self, card):
        super().on_card_rejection(card)
        assert False, "DrawPlayer should never play a card"

    def on_finish(self, winner) -> None:
        return


class RandomPlayer(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        # Choose card uniformly
        top_of_pile = pile[-1]

        can_play_l = [c for c in self.hand if c.can_play_on(top_of_pile)]
        if len(can_play_l):
            card = random.choice(can_play_l)
            self.hand.remove(card)

            if card.color == Color.WILD:
                card.color = self.on_choose_wild_color()
            return card
        else:
            return None

    def on_choose_wild_color(self):
        return random.choice([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class NoobPlayer(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        # Choose Color/Numbers -> Color/srd2 -> Type -> Wild, leaks data based on Enums
        top_of_pile = pile[-1]

        self.hand.sort(key=lambda c: c.type)
        self.hand.sort(
            key=lambda c: 5
            if c.color == Color.WILD
            else (c.color - top_of_pile.color) % 4
        )
        card = next((c for c in self.hand if c.can_play_on(top_of_pile)), None)
        if card:
            self.hand.remove(card)
            if card.color == Color.WILD:
                card.color = self.on_choose_wild_color()
        return card

    def on_choose_wild_color(self):
        colors_l = [c.color for c in self.hand if c.color != Color.WILD]
        color = Color.RED
        if len(colors_l):
            color = random.choice(colors_l)
        return color

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class BasicPlayer(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        # Choose Color/Numbers -> Color/srd2 -> numbers -> srd2 -> Wild -> Draw card
        top_of_pile = pile[-1]

        shuffled_hand = random.sample(self.hand, k=len(self.hand))

        shuffled_hand.sort(
            key=lambda c: 0
            if c.type <= Type.NINE
            else (1 if c.type < Type.CHANGECOLOR else 2)
        )
        shuffled_hand.sort(
            key=lambda c: 5
            if c.color == Color.WILD
            else (c.color - top_of_pile.color) % 4  # creating bias
        )
        card = next((c for c in shuffled_hand if c.can_play_on(top_of_pile)), None)
        if card:
            self.hand.remove(card)
            if card.color == Color.WILD:
                card.color = self.on_choose_wild_color()

        return card

    def on_choose_wild_color(self):
        return max(
            set(c.color for c in self.hand if c.color != Color.WILD),
            key=[c.color for c in self.hand if c.color != Color.WILD].count,
            default=Color.RED,
        )

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class DecentPlayer(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Color Number-single
        color_l = [
            c for c in self.hand if c.color == top_of_pile.color and c.type <= Type.NINE
        ]
        if len(color_l):
            card = random.choice(color_l)
            self.hand.remove(card)
            return card

        # Skip/Reverse/Draw2
        srd2_list = [
            c
            for c in self.hand
            if c.color == top_of_pile.color and c.type < Type.CHANGECOLOR
        ]
        if len(srd2_list):
            # TODO: Could be better or worse
            # return random.choice(srd_list)

            # Play srd with respect to other srd cards you have
            shuffled_srd2 = random.sample(srd2_list, k=len(srd2_list))
            card = max(
                set(shuffled_srd2),
                key=shuffled_srd2.count,
            )
            self.hand.remove(card)
            return card

        # Number
        if top_of_pile.type <= Type.NINE:
            type_l = [c for c in self.hand if c.type == top_of_pile.type]
            if len(type_l):
                card = random.choice(type_l)
                self.hand.remove(card)
                return card

        # Wild
        wild_l = [c for c in self.hand if c.can_play_on(top_of_pile)]
        if len(wild_l):
            card = random.choice(wild_l)
            self.hand.remove(card)
            assert card.color == Color.WILD
            card.color = self.on_choose_wild_color()

            return card

        return None

    def on_choose_wild_color(self):
        colors_l = [c.color for c in self.hand if c.color != Color.WILD]
        shuffled_colors = random.sample(colors_l, k=len(colors_l))

        return max(
            set(shuffled_colors),
            key=shuffled_colors.count,
            default=Color.RED,
        )

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class DecentPlayer2(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Shuffle
        self.hand = random.sample(self.hand, k=len(self.hand))

        type_counter = Counter([c.type for c in self.hand])
        type_count_sorted_hand = sorted(
            self.hand, key=lambda card: type_counter[card.type]
        )

        # Color Number
        # Choose card with priority to cards with less-duplicate types
        card = next(
            (
                card
                for card in reversed(type_count_sorted_hand)
                if card.color == top_of_pile.color and card.type <= Type.NINE
            ),
            None,
        )
        if card:
            self.hand.remove(card)
            return card

        # Color Skip/Reverse/Draw2
        card = next(
            (
                card
                for card in reversed(type_count_sorted_hand)
                if card.color == top_of_pile.color
                and card.type < Type.CHANGECOLOR
                and card.type > Type.NINE
            ),
            None,
        )
        if card:
            self.hand.remove(card)
            return card

        # Number
        if top_of_pile.type <= Type.NINE:
            card = next(
                (
                    card
                    for card in reversed(type_count_sorted_hand)
                    if card.type == top_of_pile.type
                ),
                None,
            )
            if card:
                self.hand.remove(card)
                return card

        # Skip/Reverse/Draw2
        if top_of_pile.type < Type.CHANGECOLOR and top_of_pile.type > Type.NINE:
            card = next(
                (
                    card
                    for card in (type_count_sorted_hand)
                    if card.type == top_of_pile.type
                ),
                None,
            )
            if card:
                self.hand.remove(card)
                return card

        # Wild
        card = next((card for card in self.hand if card.can_play_on(top_of_pile)), None)
        if card:
            self.hand.remove(card)
            assert card.color == Color.WILD

            card.color = self.on_choose_wild_color()

            return card

        return None

    def on_choose_wild_color(self):
        # Play color player has the most
        colors_l = [c.color for c in self.hand if c.color != Color.WILD]
        shuffled_colors = random.sample(colors_l, k=len(colors_l))

        return max(
            set(shuffled_colors),
            key=shuffled_colors.count,
            default=Color.RED,
        )

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class DecentPlayer3(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Shuffle
        self.hand = random.sample(self.hand, k=len(self.hand))

        type_counter = Counter([c.type for c in self.hand])
        type_count_sorted_hand = sorted(
            self.hand, key=lambda card: type_counter[card.type]
        )

        if card_counts[1] <= 2:
            # Next player is close to winning

            # Play draw 2 if we can
            card = next(
                (
                    card
                    for card in self.hand
                    if card.color == top_of_pile.color and card.type == Type.DRAW2
                ),
                None,
            )
            if not card:
                # Play draw 4 if we can
                card = next(
                    (card for card in self.hand if card.type == Type.DRAW4),
                    None,
                )
                if card:
                    assert card.color == Color.WILD
                    card.color = self.on_choose_wild_color()
            if card:
                self.hand.remove(card)
                return card

        # Color Number
        # Choose card with priority to cards with less-duplicate types
        card = next(
            (
                card
                for card in reversed(type_count_sorted_hand)
                if card.color == top_of_pile.color and card.type <= Type.NINE
            ),
            None,
        )
        if card:
            self.hand.remove(card)
            return card

        # Color Skip/Reverse/Draw2
        card = next(
            (
                card
                for card in reversed(type_count_sorted_hand)
                if card.color == top_of_pile.color
                and card.type < Type.CHANGECOLOR
                and card.type > Type.NINE
            ),
            None,
        )
        if card:
            self.hand.remove(card)
            return card

        # Number
        if top_of_pile.type <= Type.NINE:
            card = next(
                (
                    card
                    for card in reversed(type_count_sorted_hand)
                    if card.type == top_of_pile.type
                ),
                None,
            )
            if card:
                self.hand.remove(card)
                return card

        # Skip/Reverse/Draw2
        if top_of_pile.type < Type.CHANGECOLOR and top_of_pile.type > Type.NINE:
            card = next(
                (
                    card
                    for card in reversed(type_count_sorted_hand)
                    if card.type == top_of_pile.type
                ),
                None,
            )
            if card:
                self.hand.remove(card)
                return card

        # Wild
        card = next((card for card in self.hand if card.can_play_on(top_of_pile)), None)
        if card:
            self.hand.remove(card)
            assert card.color == Color.WILD
            card.color = self.on_choose_wild_color()
            return card

        return None

    def on_choose_wild_color(self):
        # Play color player has the most
        colors_l = [c.color for c in self.hand if c.color != Color.WILD]
        shuffled_colors = random.sample(colors_l, k=len(colors_l))

        return max(
            set(shuffled_colors),
            key=shuffled_colors.count,
            default=Color.RED,
        )

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


def bf_soft_streak(top_of_pile, hand):
    """Calculates the longest soft streak a player can make, this is only good for 2 players

    Args:
        pile (_type_): _description_
        hand (_type_): _description_

    Returns:
        _type_: reversed list of cards in streak
    """
    longest_seq = []

    for card_idx, card in enumerate(hand):
        if card.can_play_on(top_of_pile):
            hand_mod = copy(hand)
            hand_mod.remove(card)

            cur_seq = bf_soft_streak(card, hand_mod)
            cur_seq.append(card)
            if len(cur_seq) > len(longest_seq):
                longest_seq = cur_seq
    return longest_seq


def bf_hard_streak(top_of_pile, hand, recurse_run=False):
    """Calculates the longest hard streak a player can make, this is only good for 2 players

    Args:
        pile (_type_): _description_
        hand (_type_): _description_

    Returns:
        _type_: reversed list of cards in streak
    """
    longest_seq = []

    if recurse_run and not (
        top_of_pile.type == Type.SKIP
        or top_of_pile.type == Type.REVERSE
        or top_of_pile.type == Type.DRAW2
        or top_of_pile.type == Type.DRAW4
    ):
        return []

    for card_idx, card in enumerate(hand):
        if card.can_play_on(top_of_pile):
            hand_mod = copy(hand)
            hand_mod.remove(card)

            cur_seq = bf_hard_streak(card, hand_mod, recurse_run=True)
            cur_seq.append(card)
            if len(cur_seq) > len(longest_seq):
                longest_seq = cur_seq
    return longest_seq


class DecentPlayer4(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

    def on_turn(self, pile, card_counts, drawn):
        # lazy bug ~fix~
        card = self.pick_card(pile, card_counts, drawn)

        if card and card.color == Color.WILD:
            card.color = self.on_choose_wild_color()
        return card

    def pick_card(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Shuffle
        self.hand = random.sample(self.hand, k=len(self.hand))

        type_counter = Counter([c.type for c in self.hand])
        type_count_sorted_hand = sorted(
            self.hand, key=lambda card: type_counter[card.type]
        )

        if card_counts[1] <= 2:
            # Next player is close to winning

            # Play draw 2 if we can
            card = next(
                (
                    card
                    for card in self.hand
                    if card.color == top_of_pile.color and card.type == Type.DRAW2
                ),
                None,
            )
            if not card:
                # Play draw 4 if we can
                card = next(
                    (card for card in self.hand if card.type == Type.DRAW4),
                    None,
                )
                if card:
                    assert card.color == Color.WILD
                    card.color = self.on_choose_wild_color()
            if card:
                self.hand.remove(card)
                return card

        if len(self.hand) < 5:
            # We are close-ish to winning

            # Assuming 2p and --draw_skip
            skippers_hand = sorted(
                self.hand,
                key=lambda card: 0
                if card.type == Type.SKIP
                or card.type == Type.REVERSE
                or card.type == Type.DRAW2
                or card.type == Type.DRAW4
                else 1,
            )
            # longest_streak = bf_soft_streak(top_of_pile, skippers_hand)
            longest_streak = bf_hard_streak(top_of_pile, self.hand)
            if len(longest_streak) == len(self.hand):
                # We can hard steak to win
                card = longest_streak[-1]
                logging.debug("Hard Streak: %s" % longest_streak)

                if (
                    card.color == Color.WILD
                    and len(longest_streak) > 1
                    and longest_streak[-2].color != Color.WILD
                ):
                    # Set the color for the wildcard
                    card.color = longest_streak[-2].color
                    if card.color == Color.WILD:
                        card.color = self.on_choose_wild_color()
                self.hand.remove(card)
                return card

        # Color Number
        # Choose card with priority to cards with less-duplicate types
        card = next(
            (
                card
                for card in reversed(type_count_sorted_hand)
                if card.color == top_of_pile.color and card.type <= Type.NINE
            ),
            None,
        )
        if card:
            self.hand.remove(card)
            return card

        # Color Skip/Reverse/Draw2
        card = next(
            (
                card
                for card in reversed(type_count_sorted_hand)
                if card.color == top_of_pile.color
                and card.type < Type.CHANGECOLOR
                and card.type > Type.NINE
            ),
            None,
        )
        if card:
            self.hand.remove(card)
            return card

        # Number
        if top_of_pile.type <= Type.NINE:
            card = next(
                (
                    card
                    for card in reversed(type_count_sorted_hand)
                    if card.type == top_of_pile.type
                ),
                None,
            )
            if card:
                self.hand.remove(card)
                return card

        # Skip/Reverse/Draw2
        if top_of_pile.type < Type.CHANGECOLOR and top_of_pile.type > Type.NINE:
            card = next(
                (
                    card
                    for card in reversed(type_count_sorted_hand)
                    if card.type == top_of_pile.type
                ),
                None,
            )
            if card:
                self.hand.remove(card)
                return card

        # Wild
        card = next((card for card in self.hand if card.can_play_on(top_of_pile)), None)
        if card:
            self.hand.remove(card)
            assert card.color == Color.WILD
            card.color = self.on_choose_wild_color()
            return card

        return None

    def on_choose_wild_color(self):
        # Play color player has the most
        colors_l = [c.color for c in self.hand if c.color != Color.WILD]
        shuffled_colors = random.sample(colors_l, k=len(colors_l))

        return max(
            set(shuffled_colors),
            key=shuffled_colors.count,
            default=Color.RED,
        )

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


# TODO: Make smarter versions of 2x2 decent player
# Chaining
#   Policy for number cards you have multiple of
#   Policy for skips/reverse/draw2/draw4 chaining
# The zeros trick
# History of the game: break MDP
# Look at what the opponent is doing
# Epsilon greedy or something with bluffing
