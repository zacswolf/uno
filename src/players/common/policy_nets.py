from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from card import Card
from players.common.misc import act_filter

from players.common.action_space import ActionSpace


class PolicyNet(ABC):
    def __init__(
        self, action_space: ActionSpace, ss_size: int, args, player_idx
    ) -> None:
        super().__init__()

        self.action_space = action_space
        self.ss_size = ss_size
        self.as_size = self.action_space.size()

        self.net: torch.nn.Module

        # For saving model
        self.model_dir = args.model_dir
        self.run_name = args.run_name
        self.player_idx = player_idx

        # For loading model
        self.policy_load = args.policy_net[player_idx]

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


class PolicyNet0(PolicyNet):
    """Policy Net that doesn't check if a card is valid"""

    def __init__(
        self, action_space: ActionSpace, ss_size: int, args, player_idx
    ) -> None:
        super().__init__(action_space, ss_size, args, player_idx)

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
        return self.action_space.idx_to_card(action_idx)


class PolicyNet1(PolicyNet):
    """Policy Net that checks if a card is valid"""

    def __init__(
        self, action_space: ActionSpace, ss_size: int, args, player_idx
    ) -> None:
        super().__init__(action_space, ss_size, args, player_idx)

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
        action_idx = -1
        if np.random.random() > 0.8:
            action_idx = valid_actions_idxs[np.argmax(valid_action_dist)]
        else:
            action_idx = np.random.choice(valid_actions_idxs, p=valid_action_dist)

        # Convert to card
        return self.action_space.idx_to_card(action_idx)
