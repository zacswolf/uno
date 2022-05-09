import logging
from card import Card
from player import Player
from players.common import state_space, value_nets
from players.non_rl import BasicPlayer, DecentPlayer4


class ValueLearner(Player):
    """
    Train value net using another player's policy
    """

    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep3(game_args)

        self.policy_player = DecentPlayer4(player_args, game_args)

        self.value = value_nets.ValueNet2(self.state_space, player_args, game_args)
        self.gamma = player_args.gamma

        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def get_card(self, card: Card) -> None:
        super().get_card(card)
        self.policy_player.get_card(Card(card.type, card.color))

    def reset(self) -> None:
        super().reset()
        self.policy_player.reset()

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.state.shape[0] == self.state_space.size()
        self.state_history.append(state)

        state_val = self.value.get_value(state)
        logging.debug(f"State Value: {state_val}")

        # Get card
        card = self.policy_player.on_turn(pile, card_counts, drawn)

        # Test validity card
        if card is not None:
            assert card in self.hand
            assert card.can_play_on(top_of_pile)

        self.action_history.append(card)
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

    def on_card_rejection(self, card):
        super().on_card_rejection(card)
        self.policy_player.on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1
        self.reward_history[-1] += reward

        assert len(self.state_history) == len(self.action_history)
        assert len(self.action_history) + 1 == len(self.reward_history)

        T = len(self.state_history)

        for t, (s, a) in enumerate(zip(self.state_history, self.action_history)):
            G = sum(
                pow(self.gamma, k - t - 1) * self.reward_history[k]
                for k in range(t + 1, T + 1)
            )

            if self.update:
                self.value.update(s, G)

        # Reset for next game
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save()
        self.policy_player.on_finish(winner)
