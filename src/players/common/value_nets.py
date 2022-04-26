from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from players.common.state_space import StateSpace


class ValueNet(ABC):
    def __init__(self, state_space: StateSpace, args, player_idx) -> None:
        super().__init__()
        self.state_space = state_space
        self.ss_size = self.state_space.size()

        self.net: torch.nn.Module

        # For saving model
        self.model_dir = args.model_dir
        self.run_name = args.run_name
        self.player_idx = player_idx

        # For loading model
        self.value_load = args.value_net[player_idx]

    @abstractmethod
    def update(self, state, G: float) -> None:
        """Update value of state towards G

        Args:
            state (_type_): State that's in state_space
            G (float): Reward

        Raises:
            NotImplementedError: Must be implimented by inherited class
        """
        raise NotImplementedError()

    @abstractmethod
    def get_value(self, state) -> float:
        """Get the value of state

        Args:
            state (_type_): State that's in state_space

        Raises:
            NotImplementedError: Must be implimented by inherited class

        Returns:
            float: Value of state
        """
        raise NotImplementedError()

    def save(self, tag: str = "") -> None:
        """Save model

        Args:
            tag (str, optional): Any tags to be included in the file name. Defaults to "".
        """
        value_model_file = os.path.join(
            self.model_dir,
            f"{self.run_name}_{self.player_idx}{f'_{tag}' if tag else ''}_val.pt",
        )

        # Note: We are not saving/loading optimizer state
        torch.save(self.net.state_dict(), value_model_file)

    def load(self, path: str) -> None:
        """Load model

        Args:
            path (str): "Name of model file in models dir"
        """
        value_model_file = os.path.join(self.model_dir, path)
        self.net.load_state_dict(torch.load(value_model_file))


class ValueNet1(ValueNet):
    def __init__(self, state_space: StateSpace, args, player_idx) -> None:
        super().__init__(state_space, args, player_idx)

        n_hidden = 128

        self.net = nn.Sequential(
            nn.Linear(self.ss_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0001, betas=[0.9, 0.999]
        )
        self.loss_fnt = torch.nn.MSELoss()

        if self.value_load:
            self.load(self.value_load)

    def update(self, state, G):
        self.net.train()

        G = Variable(torch.FloatTensor([G]))
        state = Variable(torch.from_numpy(state).type(torch.float32))

        prediction = self.net(state)
        loss = self.loss_fnt(prediction, G)

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_value(self, state):
        self.net.eval()

        state = torch.from_numpy(state).type(torch.float32)
        value = self.net(state)
        return value.item()