from player import Player
from enums import Color, Type
import random


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return "random"

    def on_turn(self, pile, card_counts):
        # Choose card uniformly
        top_of_pile = pile[-1]

        can_play_l = [c for c in self.hand if c.can_play_on(top_of_pile)]
        if len(can_play_l):
            card = random.choice(can_play_l)
            self.hand.remove(card)
            return card
        else:
            return None

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color randomly
        return random.choice([Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class NoobPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return "noob"

    def on_turn(self, pile, card_counts):
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
        return card

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color uniformly from hand
        colors_l = [c.color for c in self.hand if c.color != Color.WILD]
        if len(colors_l):
            return random.choice(colors_l)
        else:
            return Color.RED

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class BasicPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return "basic"

    def on_turn(self, pile, card_counts):
        # Choose Color/Numbers -> Color/srd2 -> numbers -> srd -> Wild
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
            else (c.color - top_of_pile.color) % 4
        )
        card = next((c for c in shuffled_hand if c.can_play_on(top_of_pile)), None)
        if card:
            self.hand.remove(card)
        return card

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Play color player has the most
        return max(
            set(c.color for c in self.hand),
            key=[c.color for c in self.hand].count,
            default=Color.RED,
        )

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        return


class DecentPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def get_name(self) -> str:
        return "decent"

    def on_turn(self, pile, card_counts):
        top_of_pile = pile[-1]
        # TODO: Maybe impliment this with lambda shuffles and sorts

        # Return random card with greedy preference
        # Color Number
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
            return card

        return None

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):

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
