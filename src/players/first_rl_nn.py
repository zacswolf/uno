from copy import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from card import Card
from enums import Color, Type

from player import Player
from players.common.action_space import ASRep1
from players.common.state_space import SSRep1


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

        return x


class FirstRLPlayer(Player):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.wild_choice = None

        self.state_space = SSRep1(args)
        self.action_space = ASRep1(args)
        self.ss_size = self.state_space.size()
        self.as_size = self.action_space.size()

        self.net = PolicyNet(self.ss_size, 128, self.as_size)

        self.optimizer = torch.optim.Adam(self.net.parameters(), betas=[0.9, 0.999])

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
        action_idx = self.action_space.card_to_idx(a)

        log_prob = self.net(s)[action_idx]

        # print(prediction, G)
        # -1 * gamma_t * reward * log_prob
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def get_action(self, s):  # returns action
        self.net.eval()

        s = Variable(torch.from_numpy(s).type(torch.float32))
        action_dist = self.net(s)

        # Get action from distribution
        action_idx = np.random.choice(
            np.arange(self.as_size), p=action_dist.detach().numpy()
        )

        return self.action_space.idx_to_card(action_idx)

    def on_turn(self, pile, card_counts):

        top_of_pile = pile[-1]

        is_valid = False
        card_picked = None

        while not is_valid:
            self.wild_choice = None

            # Get state
            state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
            assert state.shape[0] == self.ss_size

            # Get card
            og_card = self.get_action(state)

            if og_card:
                assert og_card.color != Color.WILD

            # logging.info("CC: %s" % og_card)
            card = copy(og_card)

            if card and card.type >= Type.CHANGECOLOR:
                self.wild_choice = card.color
                card.color = Color.WILD

            # test card
            if card is None:
                # draw
                # okay
                is_valid = True
                card_picked = card
                self.update(state, og_card, 1, -1)
            elif card not in self.hand:
                # bad
                self.update(state, og_card, 1, -1)
            elif not card.can_play_on(top_of_pile):
                # bad
                self.update(state, og_card, 1, -1)
            else:
                # good
                is_valid = True
                card_picked = card
                self.update(state, og_card, 1, 10)
            # print("%s \tTP: %s \tCD: %s" % (valid_card, top_of_pile, card))
            # if valid_card:
            logging.info("%s \tTP: %s \tCD: %s" % (is_valid, top_of_pile, card))

        if card_picked:
            self.hand.remove(card_picked)
        return card_picked

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color randomly
        assert self.wild_choice != None
        return self.wild_choice

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        return


class SecoundRLPlayer(Player):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.wild_choice = None

        self.state_space = SSRep1(args)
        self.action_space = ASRep1(args)
        self.ss_size = self.state_space.size()
        self.as_size = self.action_space.size()

        self.net = PolicyNet(self.ss_size, 128, self.as_size)

        self.optimizer = torch.optim.Adam(self.net.parameters(), betas=[0.9, 0.999])

    def get_name(self) -> str:
        return "secrlplayer"

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
        action_idx = self.action_space.card_to_idx(a)

        log_prob = self.net(s)[action_idx]

        # print(prediction, G)
        # -1 * gamma_t * reward * log_prob
        loss = -1 * reward * log_prob

        self.optimizer.zero_grad()  # clear grad
        loss.backward()  # compute grad
        self.optimizer.step()  # apply grad

    def act_filter(self, action_tuple, top_of_pile):
        # True if can is in hand and can be played on top_of_pile
        if action_tuple[0] is None:
            return True
        else:
            # (
            #     action_tuple[0]
            #     if action_tuple[0].type < Type.CHANGECOLOR
            #     else Card(action_tuple[0].type, Color.WILD)
            # )
            card = action_tuple[0]
            if action_tuple[0].type >= Type.CHANGECOLOR:
                card = Card(action_tuple[0].type, Color.WILD)

            return card.can_play_on(top_of_pile) and card in self.hand

    def get_action(self, s, top_of_pile):  # returns action
        self.net.eval()

        s = Variable(torch.from_numpy(s).type(torch.float32))
        action_dist = self.net(s).detach().numpy()

        # Morph action dist to only include cards in our hand
        # print("hand", self.hand, "\n\n")

        actions = [
            (self.action_space.idx_to_card(action_idx), action_prob)
            for action_idx, action_prob in enumerate(action_dist)
        ]

        # Filters to only cards that are in our hand
        # valid_cards, valid_action_dist = zip(
        #     *filter(
        #         lambda action_tuple: True
        #         if action_tuple[0] is None
        #         else (
        #             action_tuple[0]
        #             if action_tuple[0].type < Type.CHANGECOLOR
        #             else Card(action_tuple[0].type, Color.WILD)
        #         )
        #         in self.hand,
        #         actions,
        #     )
        # )

        # filter to only cards in our hand and can be played on top of pile
        valid_cards, valid_action_dist = zip(
            *(
                action_tuple
                for action_tuple in actions
                if self.act_filter(action_tuple, top_of_pile)
            )
        )

        # print(valid_action_dist)
        valid_action_dist = np.asarray(valid_action_dist)
        dist_sum = np.sum(valid_action_dist)
        if dist_sum == 0:
            valid_action_dist = None
        else:
            valid_action_dist = valid_action_dist / dist_sum
        # print(valid_action_dist, "\n")
        card = np.random.choice(valid_cards, p=valid_action_dist)

        # print("valid_cards", valid_cards)
        # print("hand", self.hand)
        # print("dist", card_dist)

        # print("valid_cards", valid_cards)
        # print("hand", self.hand)
        # print("dist", card_dist)
        # raise Exception

        # Get action from distribution
        # action_idx = np.random.choice(
        #     np.arange(self.as_size), p=action_dist.detach().numpy()
        # )
        # card = self.action_space.idx_to_card(action_idx)

        in_hand = (
            True
            if card is None
            else (card if card.type < Type.CHANGECOLOR else Card(card.type, Color.WILD))
            in self.hand
        )
        # print(card)
        # print(self.hand)
        assert in_hand

        return card

    def on_turn(self, pile, card_counts):

        top_of_pile = pile[-1]

        is_valid = False
        card_picked = None

        while not is_valid:
            self.wild_choice = None

            # Get state
            state = self.state_space.get_state(self.hand, top_of_pile, card_counts)
            assert state.shape[0] == self.ss_size

            # Get card
            og_card = self.get_action(state, top_of_pile)

            if og_card:
                assert og_card.color != Color.WILD

            # logging.info("CC: %s" % og_card)
            card = copy(og_card)

            if card and card.type >= Type.CHANGECOLOR:
                self.wild_choice = card.color
                card.color = Color.WILD

            # test card
            if card is None:
                # draw
                # okay
                is_valid = True
                card_picked = card
                self.update(state, og_card, 1, -1)
            elif card not in self.hand:
                # bad
                self.update(state, og_card, 1, -1)
                assert False
            elif not card.can_play_on(top_of_pile):
                # bad
                self.update(state, og_card, 1, -1)
            else:
                # good
                is_valid = True
                card_picked = card
                self.update(state, og_card, 1, 1)
            # print("%s \tTP: %s \tCD: %s" % (valid_card, top_of_pile, card))
            # if valid_card:
            logging.info("%s \tTP: %s \tCD: %s" % (is_valid, top_of_pile, card))

        if card_picked:
            self.hand.remove(card_picked)
        return card_picked

    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, card_type):
        # Choose color randomly
        assert self.wild_choice != None
        return self.wild_choice

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Maybe hurt reward
        return
