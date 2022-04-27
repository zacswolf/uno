from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from card import Card
from players.common.misc import act_filter

from players.common.action_space import ActionSpace


class PolicyNet(ABC):
    def __init__(
        self, action_space: ActionSpace, ss_size: int, player_args, game_args
    ) -> None:
        super().__init__()

        self.action_space = action_space
        self.ss_size = ss_size
        self.as_size = self.action_space.size()

        self.net: torch.nn.Module

        # For saving model
        self.model_dir = game_args.model_dir
        self.run_name = game_args.run_name
        self.player_idx = player_args.player_idx

        # For loading model
        self.policy_load = player_args.policy_net

    @abstractmethod
    # Note: params probably wont generalize to all policy nets
    def update(self, state, action: Card | None, gamma_t: float, reward: float) -> None:
        """Update policy of choosing action while in state state

        Args:
            state (_type_): State that's in state_space
            action (Card | None): Action played on state
            gamma_t (float): Discount factor ^t
            reward (float): Reward

        Raises:
            NotImplementedError: Must be implimented by inherited class
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, hand: list[Card], state, top_of_pile: Card) -> Card | None:
        """Get action from state

        Args:
            hand (list[Card]): Player's hand
            state (_type_): State that's in state_space
            top_of_pile (Card): Card on the top of the pile

        Raises:
            NotImplementedError: Must be implimented by inherited class

        Returns:
            Card | None: Action from policy based on state
        """
        raise NotImplementedError()

    def save(self, tag: str = "") -> None:
        """Save model

        Args:
            tag (str, optional): Any tags to be included in the file name. Defaults to "".
        """
        policy_model_file = os.path.join(
            self.model_dir,
            f"{self.run_name}_{self.player_idx}{f'_{tag}' if tag else ''}_pol.pt",
        )

        # Note: We are not saving/loading optimizer state
        torch.save(self.net.state_dict(), policy_model_file)

    def load(self, path: str) -> None:
        """Load model

        Args:
            path (str): "Name of model file in models dir"
        """
        policy_model_file = os.path.join(self.model_dir, path)
        self.net.load_state_dict(torch.load(policy_model_file))


class PolNetBasic(PolicyNet):
    """Policy Net that doesn't check if a card is valid"""

    def __init__(
        self, action_space: ActionSpace, ss_size: int, player_args, game_args
    ) -> None:
        super().__init__(action_space, ss_size, player_args, game_args)

        n_hidden = 128

        self.net = nn.Sequential(
            nn.Linear(self.ss_size, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, self.as_size),
            nn.Softmax(-1),
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0001, betas=[0.9, 0.999]
        )

        if self.policy_load:
            self.load(self.policy_load)

    def update(self, state, action, gamma_t, reward):
        """
        state: state S_t
        action: action A_t
        gamma_t: gamma^t
        reward: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state = Variable(torch.from_numpy(state).type(torch.float32))
        # gamma_t = Variable(torch.FloatTensor([gamma_t]))
        reward = Variable(torch.FloatTensor([reward]))

        # Maps card to action idx
        action_idx = self.action_space.card_to_idx(action)

        log_prob = self.net(state)[action_idx]
        # -1 * gamma_t * reward * log_prob
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, hand, state, top_of_pile: Card):  # returns action
        self.net.eval()

        # Get action dist from state
        state = torch.from_numpy(state).type(torch.float32)
        action_dist = self.net(state).detach().numpy()

        # Sample
        action_idx = np.random.choice(np.arange(self.as_size), p=action_dist)

        # Convert to card
        return self.action_space.idx_to_card(action_idx, top_of_pile)


class PolNetValActions(PolicyNet):
    """Policy Net that checks if a card is valid"""

    def __init__(
        self, action_space: ActionSpace, ss_size: int, player_args, game_args
    ) -> None:
        super().__init__(action_space, ss_size, player_args, game_args)

        n_hidden = 128

        self.net = nn.Sequential(
            nn.Linear(self.ss_size, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, self.as_size),
            nn.Softmax(-1),
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0001, betas=[0.9, 0.999]
        )

        if self.policy_load:
            self.load(self.policy_load)

    def update(self, state, action, gamma_t, reward):
        """
        state: state S_t
        action: action A_t
        gamma_t: gamma^t
        reward: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state = Variable(torch.from_numpy(state).type(torch.float32))
        # gamma_t = Variable(torch.FloatTensor([gamma_t]))
        reward = Variable(torch.FloatTensor([reward]))

        # Maps card to action idx
        action_idx = self.action_space.card_to_idx(action)

        log_prob = self.net(state)[action_idx]
        # -1 * gamma_t * reward * log_prob
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, hand, state, top_of_pile: Card):  # returns action
        self.net.eval()

        # Get action dist from state
        state = torch.from_numpy(state).type(torch.float32)
        action_dist = self.net(state).detach().numpy()

        # Morph action dist to only include valid cards

        # Get valid action idxs
        assert self.as_size == action_dist.shape[0]
        valid_actions_idxs = [
            action_idx
            for action_idx in range(self.as_size)
            if act_filter(hand, self.action_space.idx_to_card(action_idx), top_of_pile)
        ]

        valid_action_dist = action_dist[valid_actions_idxs]

        # Normalize
        # TODO: Softmax this shit
        dist_sum = np.sum(valid_action_dist)
        if dist_sum == 0:
            valid_action_dist = None
        else:
            valid_action_dist = valid_action_dist / dist_sum

        # Sample
        action_idx = np.random.choice(valid_actions_idxs, p=valid_action_dist)

        # Convert to card
        return self.action_space.idx_to_card(action_idx)


class PolNetValActionsSoftmax(PolicyNet):
    """Policy Net that checks if a card is valid and does a smart softmax"""

    def __init__(
        self, action_space: ActionSpace, ss_size: int, player_args, game_args
    ) -> None:
        super().__init__(action_space, ss_size, player_args, game_args)

        n_hidden = 128

        self.net = nn.Sequential(
            nn.Linear(self.ss_size, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, self.as_size),
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0001, betas=[0.9, 0.999]
        )

        if self.policy_load:
            self.load(self.policy_load)

    def update(self, wrapped_state, action, gamma_t, reward):
        """
        wrapped_state: wrapped_state dict that contains S_t, hand_t, and top_of_pile_t
        action: action A_t
        gamma_t: gamma^t
        reward: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state = wrapped_state["state"]
        hand = wrapped_state["hand"]
        top_of_pile = wrapped_state["top_of_pile"]

        state = Variable(torch.from_numpy(state).type(torch.float32))
        # gamma_t = Variable(torch.FloatTensor([gamma_t]))
        reward = Variable(torch.FloatTensor([reward]))

        # Maps card to action idx
        action_idx = self.action_space.card_to_idx(
            action, hand=hand, top_of_pile=top_of_pile
        )

        check_card = self.action_space.idx_to_card(
            action_idx, hand=hand, top_of_pile=top_of_pile
        )
        assert (
            action == check_card
        ), f"action space is not lineing up {action} : {check_card}"

        action_vals = self.net(state)
        valid_actions_bool_mask = [
            act_filter(
                hand,
                self.action_space.idx_to_card(
                    action_idx, hand=hand, top_of_pile=top_of_pile
                ),
                top_of_pile,
            )
            for action_idx in range(self.as_size)
        ]
        action_vals[valid_actions_bool_mask] = f.softmax(
            action_vals[valid_actions_bool_mask], -1
        )
        action_vals[not valid_actions_bool_mask] = 0

        log_prob = action_vals[action_idx]
        # -1 * gamma_t * reward * log_prob
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, hand, state, top_of_pile: Card):  # returns action
        self.net.eval()

        # Get action dist from state
        state = torch.from_numpy(state).type(torch.float32)
        action_vals = self.net(state).detach()  # .numpy()

        # Morph action dist to only include valid cards

        # Get valid action idxs
        assert self.as_size == action_vals.shape[0]
        valid_actions_idxs = [
            action_idx
            for action_idx in range(self.as_size)
            if act_filter(
                hand,
                self.action_space.idx_to_card(
                    action_idx, hand=hand, top_of_pile=top_of_pile
                ),
                top_of_pile,
            )
        ]

        # Turn into a valid distribution using invalid action masking
        # https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html
        valid_action_vals = action_vals[valid_actions_idxs]

        valid_action_dist = f.softmax(valid_action_vals, -1).numpy()

        # Sample
        action_idx = np.random.choice(valid_actions_idxs, p=valid_action_dist)

        # Convert to card
        return self.action_space.idx_to_card(
            action_idx, hand=hand, top_of_pile=top_of_pile
        )
