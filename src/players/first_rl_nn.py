import logging
from enums import Color

from player import Player
from players.common import policy_nets
from players.common.action_space import ASRep1
from players.common.state_space import SSRep1


class FirstRLPlayer(Player):
    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = SSRep1(game_args)
        self.action_space = ASRep1(game_args)
        self.ss_size = self.state_space.size()
        self.as_size = self.action_space.size()

        self.policy = policy_nets.PolNetBasic(
            self.action_space, self.ss_size, player_args, game_args
        )

    def on_turn(self, pile, card_counts, drawn):
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

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        self.game_num += 1
        if self.game_num == self.num_games:
            # Save models
            self.policy.save()


class SecondRLPlayer(Player):
    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.state_space = SSRep1(game_args)
        self.action_space = ASRep1(game_args)
        self.ss_size = self.state_space.size()
        self.as_size = self.action_space.size()

        self.policy = policy_nets.PolNetValActions(
            self.action_space, self.ss_size, player_args, game_args
        )

    def on_turn(self, pile, card_counts, drawn):

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

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        self.game_num += 1
        if self.game_num == self.num_games:
            # Save models
            self.policy.save()
