import numpy as np
from card import Card
from load_args import ArgsGameShared, ArgsPlayer
from player import Player
from players.common import action_space, action_value_nets
from players.common import state_space
from players.common.state_space import State


class DeepSarsa(Player):
    """
    Algo: Deep Sarsa
    StateSpace: SSRep1
    ActionSpace: ASRep1
    PolicyNet: PolNetValActionsSoftmax

    RewardType: 1/-1 win/loss

    Paper: http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/IEEE_SSCI_2016/pdf/SSCI16_paper_338.pdf
    """

    def __init__(self, player_args: ArgsPlayer, game_args: ArgsGameShared) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update
        self.gamma = player_args.gamma

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.pred_q_net = action_value_nets.AVNetValActions(
            self.action_space, self.state_space, player_args, game_args
        )

        self.target_q_net = action_value_nets.AVNetValActions(
            self.action_space, self.state_space, player_args, game_args
        )

        hyper = player_args.hyper

        # Hyper params
        # Size of the experience replay buffer
        self.er_size = 4 if "er_size" not in hyper else hyper["er_size"]  # 1024
        # How many sars to sample and update during each iterations
        self.minibatch_size = (
            4 if "minibatch_size" not in hyper else hyper["minibatch_size"]
        )  # 32
        # How many iterations until we set the target network to the predicted network's weights
        self.num_iters_p2t = (
            16 if "num_iters_p2t" not in hyper else hyper["num_iters_p2t"]
        )  # 256

        # Cheap way to prioritize end of game
        self.final_flood = 0  # 10

        # (last_state, last_action, reward, state, action is_terminal)
        self.experience_replay = [None] * self.er_size
        self.er_idx = 0
        self.p2t_idx = 0

        assert self.minibatch_size <= self.er_size

        self.last_state = None
        self.last_action = None

    def deeq_sarsa_mini_batch(self, mini_batch_idxs: list[int]):
        mini_batch_idxs = mini_batch_idxs

        states = []
        actions = []
        targets = []

        for idx in mini_batch_idxs:
            (s, a, r, s_prime, a_prime, is_terminal) = self.experience_replay[idx]

            q_vals, _ = self.target_q_net.get_action_data(
                self.state_space.get_hand(s_prime),
                s_prime,
                self.state_space.get_top_of_pile(s_prime),
            )

            action_prime_idx = self.action_space.card_to_idx(a_prime)

            targets.append(
                r if is_terminal else r + self.gamma * q_vals[action_prime_idx]
            )

            # if not is_terminal and r:
            #     print(r)
            states.append(s)
            actions.append(a)

        self.pred_q_net.batch_update(states, actions, targets)

    def deep_sarsa_update(
        self, state: State, reward: float, action: Card | None, is_terminal: bool
    ):
        if self.update:
            # Add to experience replay
            if self.last_state is not None:
                self.experience_replay[self.er_idx % self.er_size] = (
                    self.last_state,
                    self.last_action,
                    reward,
                    state,
                    action,
                    is_terminal,
                )
                self.er_idx += 1

                if is_terminal:
                    # artificially flood (I created this)
                    for _ in range(self.final_flood):
                        self.experience_replay[self.er_idx % self.er_size] = (
                            self.last_state,
                            self.last_action,
                            reward,
                            state,
                            action,
                            is_terminal,
                        )
                        self.er_idx += 1

            # Sample
            mini_batch_idxs = None
            if self.er_idx >= self.er_size:
                # full buffer
                mini_batch_idxs = np.random.choice(
                    np.arange(self.er_size), size=self.minibatch_size, replace=False
                )

            # This is commented as I think it drives the init to a bad state due to high corelations of its data in the buffer
            # else:
            #     mini_batch = np.random.choice(
            #         np.arange(self.er_idx),
            #         size=min(self.minibatch_size, self.er_idx),
            #     )

            if mini_batch_idxs is not None:
                self.deeq_sarsa_mini_batch(mini_batch_idxs)

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
        assert state.state.shape[0] == self.state_space.size()

        reward = 0
        # artificial neg reward
        if card_counts[0] >= 9:
            reward = -(card_counts[0] - 8.0) * 0.1

        # Get card
        # Epsilon random sample
        action = self.target_q_net.get_action(self.hand, state, top_of_pile)

        # Update
        self.deep_sarsa_update(state, reward, action, False)

        # Test validity card
        if action is not None:
            assert action in self.hand
            assert action.can_play_on(top_of_pile)

        self.last_state = state
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
        reward = float(2 * win - 1)

        # if self.update and self.last_state:
        #     assert np.all(
        #         self.experience_replay[(self.er_idx - 1) % self.er_size][3].state
        #         == self.last_state.state
        #     )

        # Do an update
        self.deep_sarsa_update(self.last_state, reward, None, True)

        # Reset for next game
        self.last_state = None
        self.last_action = None
        self.target_q_net.on_finish()

        self.game_num += 1
        if self.game_num == self.num_games:
            # Save model
            self.pred_q_net.save("sarsa")
