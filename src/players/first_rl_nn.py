from copy import copy
import logging
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from card import Card
from enums import Color, Type

from player import Player
from players.common import get_ss_rep1


class PolicyNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(PolicyNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Softmax(-1),
        )

    def forward(self, x):
        x = self.model(x)

        # print(x)
        return x


class FirstRLPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.wild_choice = None

        # TODO: Fix this shit
        self.SS_SIZE = 75
        self.A_SIZE = 15 * 4 + 1  # plus one for the draw/noop

        self.net = PolicyNet(self.SS_SIZE, 128, self.A_SIZE)

        self.optimizer = torch.optim.Adam(self.net.parameters(), betas=[0.9, 0.999])

        self.was_card_rejected = False

    def get_name(self) -> str:
        return "firstrlplayer"

    def update(self, s, a: Card | None, gamma_t, reward):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
            1
        reward: G-v(S_t,w) or just G or just R'
            -1 if card is invalid for any reason
            1 of card is valid
        """

        self.net.train()

        s = Variable(torch.from_numpy(s).type(torch.float32))
        # gamma_t = Variable(torch.FloatTensor([gamma_t]))
        reward = Variable(torch.FloatTensor([reward]))

        # Maps card to action idx
        action_idx = None
        if not a:
            action_idx = self.A_SIZE - 1  # Drawing/Noop
        else:
            assert a.color != Color.WILD
            action_idx = a.color * (Type.DRAW4 + 1) + a.type

        log_prob = self.net(s)[action_idx]

        # print(prediction, G)
        # -1 * gamma_t * reward * log_prob
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def eval(self, s):  # returns action

        self.net.eval()

        s = Variable(torch.from_numpy(s).type(torch.float32))
        r = self.net(s)

        action_idx = np.random.choice(np.arange(self.A_SIZE), p=r.detach().numpy())
        # logging.info("act_idx %s" % action_idx)
        # Maps action idx to card
        if action_idx != self.A_SIZE - 1:
            color = action_idx // (Type.DRAW4 + 1)
            c_type = action_idx % (Type.DRAW4 + 1)
            c = Card(Type(c_type), Color(color))
            # logging.info("act_c %s" % c)
            return c
        return None

    def on_turn(self, pile, card_counts):

        top_of_pile = pile[-1]

        valid_card = False
        while not valid_card:
            self.wild_choice = None
            # get statespace
            state = get_ss_rep1(self.hand, top_of_pile, card_counts)
            assert state.shape[0] == self.SS_SIZE

            # get card
            og_card = self.eval(state)
            # logging.info("CC: %s" % og_card)
            card = copy(og_card)
            if card and card.type >= Type.CHANGECOLOR:
                self.wild_choice = card.color
                card.color = Color.WILD

            # test card
            if card is None:
                # draw
                # okay
                valid_card = True
                self.update(state, og_card, 1, -1)
            elif card not in self.hand:
                # bad
                self.update(state, og_card, 1, -1)
            elif not card.can_play_on(top_of_pile):
                # bad
                self.update(state, og_card, 1, -1)
            else:
                valid_card = True
                # good
                self.update(state, og_card, 1, 10)
            # print("%s \tTP: %s \tCD: %s" % (valid_card, top_of_pile, card))
            # if valid_card:
            logging.info("%s \tTP: %s \tCD: %s" % (bool(valid_card), top_of_pile, card))

        return card

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color randomly
        assert self.wild_choice != None
        return self.wild_choice

    def on_card_rejection(self, card):
        super().on_card_rejection(card)
        # Hurt reward
        self.was_card_rejected = True

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        return
