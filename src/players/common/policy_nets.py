from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torch.autograd import Variable
from card import Card
from load_args import ArgsGameShared, ArgsPlayer
from players.common import sampler
from players.common import state_space
from players.common.misc import val_action_mask

from players.common.action_space import ActionSpace
from players.common.state_space import State, StateSpace


class PolicyNet(ABC):
    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        player_args: ArgsPlayer,
        game_args: ArgsGameShared,
    ) -> None:
        super().__init__()

        self.action_space = action_space
        self.state_space = state_space
        self.as_size = self.action_space.size()
        self.ss_size = self.state_space.size()
        self.loss_vals=[]

        self.net: torch.nn.Module

        # For saving model
        self.model_dir = game_args.model_dir
        self.run_name = game_args.run_name
        self.player_idx = player_args.player_idx

        # For loading model
        self.policy_load = player_args.policy_net

    @abstractmethod
    # Note: params probably wont generalize to all policy nets
    def update(self, state: State, action: Card | None, reward: float) -> None:
        """Update policy of choosing action while in state state

        Args:
            state (State): State that's in state_space
            action (Card | None): Action played on state
            reward (float): Reward

        Raises:
            NotImplementedError: Must be implimented by inherited class
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action(
        self, hand: list[Card], state: State, top_of_pile: Card
    ) -> Card | None:
        """Get action from state

        Args:
            hand (list[Card]): Player's hand
            state (State): State that's in state_space
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
        policy_model_file = None
        if self.policy_load:
            # Saves to same file loaded in
            policy_model_file = os.path.join(self.model_dir, self.policy_load)
            plt.title(self.value_load + " policy model loss")
        else:
            policy_model_file = os.path.join(
                self.model_dir,
                f"{self.run_name}_{self.player_idx}{f'_{tag}' if tag else ''}_pol.pt",
            )
            plt.title(f"{self.run_name}_{self.player_idx}{f'_{tag}' if tag else ''}_pol.pt policy model loss")

        plt.plot(np.arange(len(self.loss_vals)), self.loss_vals, 'o--')
        plt.show()

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
    """Policy Net that doesn't check if a card is valid

    Don't use this
    """

    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        player_args: ArgsPlayer,
        game_args: ArgsGameShared,
    ) -> None:
        super().__init__(action_space, state_space, player_args, game_args)

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
            nn.Linear(n_hidden, self.as_size),
            nn.Softmax(-1),
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0001, betas=[0.9, 0.999]
        )

        # Choose sampler
        sampler_str = player_args.sampler if player_args.sampler else "efd"
        assert sampler_str == "efd" or sampler_str == "efdnd", "Invalid sampler"
        self.sampler = sampler.from_sampler_str(sampler_str)

        if self.policy_load:
            self.load(self.policy_load)

    def update(self, state, action, reward):
        """
        state: state S_t
        action: action A_t
        reward: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state_torch = Variable(torch.from_numpy(state.state).type(torch.float32))
        reward = Variable(torch.FloatTensor([reward]))

        # Maps card to action idx
        action_idx = self.action_space.card_to_idx(action)

        log_prob = self.net(state_torch)[action_idx]
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, hand, state, top_of_pile: Card):  # returns action
        self.net.eval()

        # Get action dist from state
        state_torch = torch.from_numpy(state.state).type(torch.float32)
        action_dist = self.net(state_torch).detach().numpy()

        # Don't mask
        val_actions_mask = np.full(action_dist.shape, True)

        # Sample
        action_idx = self.sampler(action_dist, val_actions_mask)

        # Convert to card
        return self.action_space.idx_to_card(action_idx, top_of_pile)


class PolNetValActions(PolicyNet):
    """Policy Net that checks if a card is valid"""

    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        player_args: ArgsPlayer,
        game_args: ArgsGameShared,
    ) -> None:
        super().__init__(action_space, state_space, player_args, game_args)

        self.epsilon = player_args.epsilon
        n_hidden = 128

        self.game_loss_vals=[]

        self.net = nn.Sequential(
            nn.Linear(self.ss_size, n_hidden),
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

        # Choose sampler
        sampler_str = player_args.sampler if player_args.sampler else "efd"
        assert sampler_str == "efd" or sampler_str == "efdnd", "Invalid sampler"
        self.sampler = sampler.from_sampler_str(sampler_str)

        if self.policy_load:
            self.load(self.policy_load)

    def update(self, state, action, reward):
        """
        state: state S_t
        action: action A_t
        reward: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state_torch = Variable(torch.from_numpy(state.state).type(torch.float32))
        reward = Variable(torch.FloatTensor([reward]))

        # Maps card to action idx
        action_idx = self.action_space.card_to_idx(action)

        log_prob = self.net(state_torch)[action_idx]
        loss = -1 * reward * log_prob
        self.game_loss_vals.append(loss.item())

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, hand, state, top_of_pile: Card):  # returns action
        self.net.eval()

        # Get action dist from state
        state_torch = torch.from_numpy(state.state).type(torch.float32)
        action_dist = self.net(state_torch).detach().numpy()

        # Morph action dist to only include valid cards

        # Get valid action idxs
        assert self.as_size == action_dist.shape[0]

        val_actions_mask = val_action_mask(hand, top_of_pile, self.action_space)
        valid_action_dist = action_dist[val_actions_mask]

        # Normalize
        dist_sum = float(np.sum(valid_action_dist))
        if dist_sum == 0.0:
            action_dist = np.ones(action_dist.shape)
            action_dist /= np.linalg.norm(action_dist)
        else:
            action_dist /= dist_sum

        # Sample epsilon soft but bc its already softmaxed its just e-greedy
        action_idx = self.sampler(action_dist, val_actions_mask, self.epsilon)

        # Convert to card
        return self.action_space.idx_to_card(action_idx)
    
    def on_finish(self):
        self.loss_vals.append(np.average(self.game_loss_vals))
        self.game_loss_vals=[]


class PolNetValActionsSoftmax(PolicyNet):
    """Policy Net that checks if a card is valid and does a smart softmax"""

    def __init__(
        self,
        action_space: ActionSpace,
        state_space: StateSpace,
        player_args: ArgsPlayer,
        game_args: ArgsGameShared,
    ) -> None:
        super().__init__(action_space, state_space, player_args, game_args)

        n_hidden = 128
        self.epsilon = player_args.epsilon

        self.game_loss_vals=[]

        self.net = nn.Sequential(
            nn.Linear(self.ss_size, n_hidden),
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

        # Choose sampler
        sampler_str = player_args.sampler if player_args.sampler else "es"
        assert sampler_str != "efd" and sampler_str != "efdnd", "Invalid Sampler"
        self.sampler = sampler.from_sampler_str(sampler_str)

        if self.policy_load:
            self.load(self.policy_load)

    def update(self, state, action, reward):
        """
        state: state that can create S_t, hand_t, and top_of_pile_t
        action: action A_t
        reward: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state_np = state.state
        hand = self.state_space.get_hand(state)
        top_of_pile = self.state_space.get_top_of_pile(state)

        state_torch = Variable(torch.from_numpy(state_np).type(torch.float32))
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

        action_vals = self.net(state_torch)
        val_actions_mask = val_action_mask(hand, top_of_pile, self.action_space)
        action_vals[val_actions_mask] = f.softmax(action_vals[val_actions_mask], -1)
        action_vals[np.logical_not(val_actions_mask)] = 0

        log_prob = action_vals[action_idx]
        loss = -1 * reward * log_prob
        self.game_loss_vals.append(loss.item())

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, hand, state, top_of_pile: Card):  # returns action
        self.net.eval()

        # Get action dist from state
        state_torch = torch.from_numpy(state.state).type(torch.float32)
        action_vals = self.net(state_torch).detach().numpy()
        assert self.as_size == action_vals.shape[0]

        # Turn into a valid distribution using invalid action masking
        # https://costa.sh/blog-a-closer-look-at-invalid-action-masking-in-policy-gradient-algorithms.html

        val_actions_mask = val_action_mask(hand, top_of_pile, self.action_space)

        # Sample epsilon soft
        action_idx = self.sampler(action_vals, val_actions_mask, self.epsilon)

        # Convert to card
        return self.action_space.idx_to_card(
            action_idx, hand=hand, top_of_pile=top_of_pile
        )

    def on_finish(self):
        self.loss_vals.append(np.average(self.game_loss_vals))
        self.game_loss_vals=[]