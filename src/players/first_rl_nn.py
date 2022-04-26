import logging
from enums import Color

from player import Player
from players.common import policy_nets
from players.common.action_space import ASRep1
from players.common.state_space import SSRep1


class FirstRLPlayer(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

        self.state_space = SSRep1(args)
        self.action_space = ASRep1(args)
        self.ss_size = self.state_space.size()
        self.as_size = self.action_space.size()

        self.policy = policy_nets.PolicyNet0(
            self.action_space, self.state_space.size(), args, player_idx
        )

    def get_name(self) -> str:
        return "firstrlplayer"

    def on_turn(self, pile, card_counts):
        top_of_pile = pile[-1]

        is_valid = False
        card_picked = None

        while not is_valid:

            # Get state
            state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
            assert state.shape[0] == self.ss_size

            # Get card
            card = self.policy.get_action(self.hand, state, top_of_pile)

            if card:
                assert card.color != Color.WILD

            # test card
            if card is None:
                # draw
                # okay
                is_valid = True
                card_picked = card
                self.policy.update(state, card, 1, -1)
            elif card not in self.hand:
                # bad
                self.policy.update(state, card, 1, -1)
            elif not card.can_play_on(top_of_pile):
                # bad
                self.policy.update(state, card, 1, -1)
            else:
                # good
                is_valid = True
                card_picked = card
                self.policy.update(state, card, 1, 10)
            # print("%s \tTP: %s \tCD: %s" % (valid_card, top_of_pile, card))
            # if valid_card:
            logging.info(f"{is_valid} \tTP: {top_of_pile} \tCD: {card}")

        if card_picked:
            self.hand.remove(card_picked)
        return card_picked

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color randomly
        assert False, "This model should always color the wild card on turn"

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        return


class SecondRLPlayer(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)
        self.wild_choice = None

        self.state_space = SSRep1(args)
        self.action_space = ASRep1(args)
        self.ss_size = self.state_space.size()
        self.as_size = self.action_space.size()

        self.policy = policy_nets.PolicyNet1(
            self.action_space, self.state_space.size(), args, player_idx
        )

    def get_name(self) -> str:
        return "secrlplayer"

    def on_turn(self, pile, card_counts):

        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.shape[0] == self.ss_size

        # Get card
        card = self.policy.get_action(self.hand, state, top_of_pile)

        if card:
            assert card.color != Color.WILD

        # test card
        if card is None:
            # draw
            # okay
            self.policy.update(state, card, 1, -1)
        elif card not in self.hand:
            # bad
            # self.policy.update(state, card, 1, -1)
            assert False, "Our policy filters out invalid actions"
        elif not card.can_play_on(top_of_pile):
            # bad
            # self.policy.update(state, card, 1, -1)
            assert False, "Our policy filters out invalid actions"
        else:
            # good
            self.policy.update(state, card, 1, 1)
        # if valid_card:
        logging.info(f"TP: {top_of_pile} \tCD: {card}")

        if card:
            self.hand.remove(card)
        return card

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color randomly
        assert False, "This model should always color the wild card on turn"

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        return