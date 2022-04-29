from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from card import Card
from load_args import ArgsGameShared, ArgsPlayer
from players.common import sampler
from players.common.misc import val_action_mask

from players.common.action_space import ActionSpace


class ActionValueNet(ABC):
    def __init__(
        self,
        action_space: ActionSpace,
        ss_size: int,
        player_args: ArgsPlayer,
        game_args: ArgsGameShared,
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
    def update(
        self,
        state,
        action: Card | None,
        target: float,
        action_validity_mask: np.ndarray | None = None,
    ) -> None:
        """Update policy of choosing action while in state state

        Args:
            state (_type_): State that's in state_space
            action (Card | None): Action played on state
            target (float): Target G or estimate of G where G is the cumulative reward

        Raises:
            NotImplementedError: Must be implimented by inherited class
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action_data(
        self, hand: list[Card], state, top_of_pile: Card
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get action data from state

        Examples of Action data are action values for a Q function or Action probs for a policy

        Args:
            hand (list[Card]): Player's hand
            state (_type_): State that's in state_space
            top_of_pile (Card): Card on the top of the pile

        Raises:
            NotImplementedError: Must be implimented by inherited class

        Returns:
            (np.ndarray, np.ndarray): Action data based on state and a boolean action validity mask
        """
        raise NotImplementedError()

    def get_action(self, hand: list[Card], state, top_of_pile: Card) -> Card | None:
        """Get action from state

        Args:
            hand (list[Card]): Player's hand
            state (_type_): State that's in state_space
            top_of_pile (Card): Card on the top of the pile

        Raises:
            NotImplementedError: Must be implimented by inherited class

        Returns:
            Card | None: Action from action data based on state
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
        else:
            policy_model_file = os.path.join(
                self.model_dir,
                f"{self.run_name}_{self.player_idx}{f'_{tag}' if tag else ''}_av.pt",
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


class AVNetValActions(ActionValueNet):
    """Action Value Net that checks if a card is valid"""

    def __init__(
        self, action_space: ActionSpace, ss_size: int, player_args, game_args
    ) -> None:
        super().__init__(action_space, ss_size, player_args, game_args)

        n_hidden = 128
        self.epsilon = player_args.epsilon

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

    def update(self, state, action, target, action_validity_mask=None):
        """
        state: state S_t
        action: action A_t
        target: G-v(S_t,w) or just G or just R'
        """

        self.net.train()

        state = Variable(torch.from_numpy(state).type(torch.float32))
        target = Variable(torch.FloatTensor([target]))

        # Maps card to action idx
        action_idx = self.action_space.card_to_idx(action)

        if action_validity_mask is not None:
            assert action_validity_mask[action_idx]

        prediction = self.net(state)[action_idx]
        loss = torch.square(target - prediction)

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action_data(self, hand, state, top_of_pile: Card):
        # The action data are the Q values
        self.net.eval()

        # Get action vals from state
        state = torch.from_numpy(state).type(torch.float32)
        action_vals = self.net(state).detach().numpy()

        # Get valid action mask
        assert self.as_size == action_vals.shape[0]
        val_actions_mask = val_action_mask(hand, top_of_pile, self.action_space)

        return (action_vals, val_actions_mask)

    def get_action(self, hand, state, top_of_pile: Card):
        self.net.eval()
        (action_vals, val_actions_mask) = self.get_action_data(hand, state, top_of_pile)

        # e-greedy
        action_idx = sampler.epsilon_greedy_sample(
            action_vals, val_actions_mask, self.epsilon
        )

        return self.action_space.idx_to_card(action_idx, hand, top_of_pile=top_of_pile)
