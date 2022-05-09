import copy
import logging
from load_args import ArgsGameShared, ArgsPlayer
from player import Player
from players.common import action_space, state_space, policy_nets, value_nets


class OneStepActorCritic(Player):
    def __init__(self, player_args: ArgsPlayer, game_args: ArgsGameShared) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.policy = policy_nets.PolNetValActions(
            self.action_space, self.state_space, player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)
        self.gamma = player_args.gamma

        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_action = None

    def ac_update(self, reward, state_prime_value):
        if self.update:
            if self.last_state is not None:
                delta = reward + self.gamma * state_prime_value - self.last_state_value

                # Update value net
                self.value.update(self.last_state, delta)

                # Update policy net
                self.policy.update(self.last_state, self.last_action, self.I * delta)

                self.I *= self.gamma  # might be one off

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.state.shape[0] == self.state_space.size()
        state_value = self.value.get_value(state)
        logging.debug(f"state value: {state}")

        reward = 0

        self.ac_update(reward, state_value)

        # Get card
        action = self.policy.get_action(self.hand, state, top_of_pile)

        # Test validity card
        if action is not None:
            assert action in self.hand
            assert action.can_play_on(top_of_pile)

        self.last_state = state
        self.last_state_value = state_value
        self.last_action = action

        # Return card
        if action:
            self.hand.remove(action)
        return action

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1

        if self.last_state is not None:
            # delta = self.last_reward + self.gamma * reward - self.last_state_value
            delta = reward - self.last_state_value

            # Update value net
            self.value.update(self.last_state, delta)

            # Update policy net
            self.policy.update(self.last_state, self.last_action, delta)

        # Reset for next game
        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_action = None
        self.value.on_finish()
        self.policy.on_finish()

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save("ac")
            self.policy.save("ac")


class OneStepActorCriticSoft(Player):
    def __init__(self, player_args: ArgsPlayer, game_args: ArgsGameShared) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        # self.policy is the difference between OneStepActorCritic and OneStepActorCriticSoft
        self.policy = policy_nets.PolNetValActionsSoftmax(
            self.action_space, self.state_space, player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)
        self.gamma = player_args.gamma

        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_action = None

    def ac_update(self, reward: float, state_value):
        if self.update:
            if self.last_state is not None:
                delta = reward + self.gamma * state_value - self.last_state_value

                # Update value net
                self.value.update(self.last_state, delta)

                # Update policy net
                self.policy.update(self.last_state, self.last_action, self.I * delta)

                self.I *= self.gamma  # might be one off

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.state.shape[0] == self.state_space.size()
        state_value = self.value.get_value(state)
        logging.debug(f"state value: {state_value}")

        reward = 0

        self.ac_update(reward, state_value)

        # Get card
        action = self.policy.get_action(self.hand, state, top_of_pile)

        # Test validity card
        if action is not None:
            assert action in self.hand
            assert action.can_play_on(top_of_pile)

        self.last_state = state
        self.last_state_value = state_value
        self.last_action = action

        # Return card
        if action:
            self.hand.remove(action)
        return action

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1

        # Update
        self.ac_update(reward, 0.0)

        # Reset for next game
        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_action = None
        self.value.on_finish()
        self.policy.on_finish()

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save("acsoft")
            self.policy.save("acsoft")
