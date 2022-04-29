import copy
import numpy as np
from card import Card
from player import Player
from players.common import action_space, action_value_nets, state_space


class QLearner(Player):
    """
    Algo: Deep Q-learning with Experience Replay
    StateSpace: SSRep1
    ActionSpace: ASRep1
    ValueNet: ValueNet1
    PolicyNet: PolNetValActionsSoftmax

    RewardType: 1/-1 win/loss

    Paper: https://arxiv.org/pdf/1312.5602.pdf
    """

    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update
        self.gamma = player_args.gamma

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.pred_q_net = action_value_nets.AVNetValActions(
            self.action_space, self.state_space.size(), player_args, game_args
        )

        self.target_q_net = action_value_nets.AVNetValActions(
            self.action_space, self.state_space.size(), player_args, game_args
        )

        # Hyper params
        # Size of the experience replay buffer
        self.er_size = 256
        # How many sars to sample and update during each iterations
        self.minibatch_size = 16
        # How many iterations until we set the target network to the predicted network's weights
        self.num_iters_p2t = 32

        # (wrapped_last_state, last_action, reward, wrapped_state, is_terminal)
        self.experience_replay = [None] * self.er_size
        self.er_idx = 0
        self.p2t_idx = 0

        assert self.minibatch_size < self.er_size

        self.last_wrapped_state = None
        self.last_action = None

    def deep_q_update(self, wrapped_state, reward, is_terminal):
        if self.update:
            # Add to experience replay
            if self.last_wrapped_state is not None:
                self.experience_replay[self.er_idx % self.er_size] = (
                    self.last_wrapped_state,
                    self.last_action,
                    reward,
                    wrapped_state,
                    is_terminal,
                )
                self.er_idx += 1

            # Sample
            mini_batch = []
            if self.er_idx >= self.er_size:
                # full buffer
                mini_batch = np.random.choice(
                    np.arange(self.er_size), size=self.minibatch_size
                )

            # This is commented as I think it drives the init to a bad state due to high corelations of its data in the buffer
            # else:
            #     mini_batch = np.random.choice(
            #         np.arange(self.er_idx),
            #         size=min(self.minibatch_size, self.er_idx),
            #     )

            # Update mini_batch
            for idx in mini_batch:
                (
                    ws,
                    a,
                    r,
                    ws_prime,
                    is_terminal,
                ) = self.experience_replay[idx]

                _, val_mask = self.target_q_net.get_action_data(
                    ws["hand"], ws["state"], ws["top_of_pile"]
                )

                q_vals, _ = self.target_q_net.get_action_data(
                    ws_prime["hand"], ws_prime["state"], ws_prime["top_of_pile"]
                )
                la = self.action_space.card_to_idx(a)
                assert val_mask[self.action_space.card_to_idx(a)]

                target = r if is_terminal else r + self.gamma * np.max(q_vals[val_mask])

                # TODO: Not actually batching here
                self.pred_q_net.update(
                    ws["state"], a, target, action_validity_mask=val_mask
                )

            # Check to see if we need to transfer pred into target
            self.p2t_idx += 1
            if self.p2t_idx == self.num_iters_p2t:
                self.p2t_idx = 0
                # Copy predition net's weights into the target net

                self.target_q_net.net.load_state_dict(
                    self.pred_q_net.net.state_dict(), strict=True
                )

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state
        state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state.shape[0] == self.state_space.size()

        hand_dc = copy.deepcopy(
            self.hand
        )  # [Card(card.type, card.color) for card in self.hand]
        wrapped_state = {
            "state": state,
            "hand": hand_dc,
            "top_of_pile": Card(top_of_pile.type, top_of_pile.color),  # copy
        }

        reward = 0
        # artificial neg reward
        if card_counts[0] >= 9:
            reward = -(card_counts[0] - 8.0) * 0.1

        # TODO:Update
        self.deep_q_update(wrapped_state, reward, False)

        # Get card
        # Epsilon random sample
        action = self.target_q_net.get_action(self.hand, state, top_of_pile)

        # Test validity card
        if action is not None:
            assert action in self.hand
            assert action.can_play_on(top_of_pile)

        self.last_wrapped_state = wrapped_state
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

        # Do an update
        self.deep_q_update(self.last_wrapped_state, reward, True)

        # Reset for next game
        self.last_wrapped_state = None
        self.last_action = None

        self.game_num += 1
        if self.game_num == self.num_games:
            # Save models
            self.pred_q_net.save()
