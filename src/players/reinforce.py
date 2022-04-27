import logging
from player import Player
from players.common import policy_nets, value_nets
from players.common.action_space import ASRep1
from players.common.state_space import SSRep1
from enums import Color


class Reinforce1(Player):
    def __init__(self, player_idx, args) -> None:
        super().__init__(player_idx, args)

        self.num_games = args.num_games
        self.game_num = 0

        self.state_space = SSRep1(args)
        self.action_space = ASRep1(args)

        self.policy = policy_nets.PolicyNet1(
            self.action_space, self.state_space.size(), args, player_idx
        )
        self.value = value_nets.ValueNet1(self.state_space, args, player_idx)

        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def get_name(self) -> str:
        return "reinforce1"

    def on_turn(self, pile, card_counts):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.shape[0] == self.state_space.size()
        self.state_history.append(state)

        # Get card
        card = self.policy.get_action(self.hand, state, top_of_pile)

        # Test validity card
        if card is not None:
            assert card in self.hand
            assert card.color != Color.WILD
            assert card.can_play_on(top_of_pile)
            self.action_history.append(card)
        else:
            self.action_history.append(None)

        assert len(self.state_history) == len(self.action_history)

        # Save Reward

        # sketchy way to keep it from drawing too much
        # reward = 0
        # if card is None:
        #     reward = -1.0 * card_counts[0]
        self.reward_history.append(0)

        # Return card
        if card:
            self.hand.remove(card)
        return card

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        assert False, "This model should always color the wild card on turn"

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1
        self.reward_history[-1] += reward

        gamma = 1

        assert len(self.state_history) == len(self.action_history)
        assert len(self.action_history) + 1 == len(self.reward_history)

        T = len(self.state_history)

        for t, (s, a) in enumerate(zip(self.state_history, self.action_history)):
            # G = reward
            # G = sum(pow(gamma, k - t - 1) * self.reward_history[k] for k in range(t + 1, T + 1))
            G = sum(self.reward_history[k] for k in range(t + 1, T + 1))
            delta = G - self.value.get_value(s)
            self.value.update(s, G)
            self.policy.update(s, a, pow(gamma, t), delta)

        # Reset for next game
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save()
            self.policy.save()
