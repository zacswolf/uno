import copy
import logging
from player import Player
from players.common import action_space, state_space, policy_nets, value_nets


class ReinforceValActions(Player):
    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.policy = policy_nets.PolNetValActions(
            self.action_space, self.state_space.size(), player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)

        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def on_turn(self, pile, card_counts, drawn):
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
            if self.update:
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


class ReinforceValActionsSoftmax(Player):
    """
    Algo: Reinforce
    StateSpace: SSRep1
    ActionSpace: ASRep1
    ValueNet: ValueNet1
    PolicyNet: PolNetValActionsSoftmax

    RewardType: 1/-1 win/loss
    """

    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.policy = policy_nets.PolNetValActionsSoftmax(
            self.action_space, self.state_space.size(), player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)

        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.shape[0] == self.state_space.size()

        wrapped_state = {
            "state": state,
            "hand": copy.deepcopy(self.hand),
            "top_of_pile": copy.copy(top_of_pile),
        }
        self.state_history.append(wrapped_state)

        # Get card
        card = self.policy.get_action(self.hand, state, top_of_pile)

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

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1
        self.reward_history[-1] += reward

        gamma = 1

        assert len(self.state_history) == len(self.action_history)
        assert len(self.action_history) + 1 == len(self.reward_history)

        T = len(self.state_history)

        for t, (ws, a) in enumerate(zip(self.state_history, self.action_history)):
            # G = reward
            # G = sum(pow(gamma, k - t - 1) * self.reward_history[k] for k in range(t + 1, T + 1))
            G = sum(self.reward_history[k] for k in range(t + 1, T + 1))

            s = ws["state"]
            delta = G - self.value.get_value(s)
            if self.update:
                self.value.update(s, G)
                self.policy.update(ws, a, pow(gamma, t), delta)

        # Reset for next game
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save()
            self.policy.save()


class ReinforceValActionsSoftmax2(Player):
    """
    Algo: Reinforce
    StateSpace: SSRep2
    ActionSpace: ASRep2
    ValueNet: ValueNet1
    PolicyNet: PolNetValActionsSoftmax

    RewardType: 1/-1 win/loss
    """

    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep2(game_args)
        self.action_space = action_space.ASRep2(game_args)

        self.policy = policy_nets.PolNetValActionsSoftmax(
            self.action_space, self.state_space.size(), player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)

        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.shape[0] == self.state_space.size()

        wrapped_state = {
            "state": state,
            "hand": copy.deepcopy(self.hand),
            "top_of_pile": copy.copy(top_of_pile),
        }
        self.state_history.append(wrapped_state)

        # Get card
        card = self.policy.get_action(self.hand, state, top_of_pile)

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

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1
        self.reward_history[-1] += reward

        gamma = 1

        assert len(self.state_history) == len(self.action_history)
        assert len(self.action_history) + 1 == len(self.reward_history)

        T = len(self.state_history)

        for t, (ws, a) in enumerate(zip(self.state_history, self.action_history)):
            # G = reward
            # G = sum(pow(gamma, k - t - 1) * self.reward_history[k] for k in range(t + 1, T + 1))
            G = sum(self.reward_history[k] for k in range(t + 1, T + 1))

            s = ws["state"]
            delta = G - self.value.get_value(s)
            if self.update:
                self.value.update(s, G)
                self.policy.update(ws, a, pow(gamma, t), delta)

        # Reset for next game
        self.state_history = []
        self.action_history = []
        self.reward_history = [0]

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save()
            self.policy.save()
