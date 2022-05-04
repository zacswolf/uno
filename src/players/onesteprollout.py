import copy
from doctest import SKIP
import logging
from player import Player
from players.common import action_space, state_space, value_nets
from enums import Color, Type
import numpy as np
from card import Card


class OneStepRollout(Player):
    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)
        self.gamma = player_args.gamma

        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_reward = 0
        self.last_action = None

    def act_filter(self, hand, card: Card | None, top_of_pile: Card):
        # True if can is in hand and can be played on top_of_pile
        if card is None:
            return True
        else:
            return card.can_play_on(top_of_pile) and card in hand

    def get_hand_copy(self):
        return copy.deepcopy(self.hand)

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state prime
        state_prime = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state_prime.state.shape[0] == self.state_space.size()
        state_prime_value = self.value.get_value(state_prime)
        logging.debug(f"state prime value: {state_prime_value}")

        # Get card
        action_prime = None
        temp_card_count = np.array(copy.deepcopy(card_counts))
        temp_card_count[0] -= 1
        temp_card_count0 = np.array(copy.deepcopy(temp_card_count))
        temp_card_count1 = np.array(copy.deepcopy(temp_card_count))
        temp_card_count0[1:] += 1
        temp_card_count1[1:] -= 1

        valid_actions_idxs = [
            action_idx
            for action_idx in range(len(self.hand))
            if self.act_filter(self.hand, self.hand[action_idx], top_of_pile)
        ]

        next_card_values = []
        colors = []
        for idx in valid_actions_idxs:
            card = self.hand[idx]
            temp_hand = self.get_hand_copy()
            temp_hand.remove(card)
            temp_card = copy.deepcopy(card)
            temp_next_card_values = []
            if card.type >= Type.CHANGECOLOR:
                for c in range(Color.WILD):
                    temp_card.color = Color(c)
                    if card.type == Type.DRAW4:
                        temp_card_count[1] += 4
                        temp_state = self.state_space.get_state(
                            temp_hand, temp_card, temp_card_count
                        )
                        temp_next_card_values.append(self.value.get_value(temp_state))
                    else:
                        temp_state = self.state_space.get_state(
                            temp_hand, temp_card, temp_card_count
                        )
                        temp_state0 = self.state_space.get_state(
                            temp_hand, temp_card, temp_card_count0
                        )
                        temp_state1 = self.state_space.get_state(
                            temp_hand, temp_card, temp_card_count1
                        )
                        temp_value = self.value.get_value(temp_state)
                        temp_value0 = self.value.get_value(temp_state0)
                        temp_value1 = self.value.get_value(temp_state1)
                        chance_not_playable_card = 1 - (
                            (1 / Color.WILD) + 8 / 108
                        )  # 8/108 is for Wilds
                        chance_opp_has_card = 1 - (
                            chance_not_playable_card ** card_counts[1]
                        )
                        temp_next_card_values.append(
                            chance_opp_has_card * temp_value1
                            + (1 - chance_opp_has_card)
                            * chance_not_playable_card
                            * temp_value0
                            + (1 - chance_opp_has_card)
                            * (1 - chance_not_playable_card)
                            * temp_value
                        )
                next_card_values.append(max(temp_next_card_values))
                colors.append(np.argmax(temp_next_card_values))
            elif card.type == Type.DRAW2:
                temp_card_count[1] += 2
                temp_state = self.state_space.get_state(
                    temp_hand, temp_card, temp_card_count
                )
                next_card_values.append(self.value.get_value(temp_state))
                colors.append(card.color)
            elif card.type == Type.SKIP:
                temp_state = self.state_space.get_state(
                    temp_hand, temp_card, temp_card_count
                )
                next_card_values.append(self.value.get_value(temp_state))
                colors.append(card.color)
            else:
                # Approximation that it's the same color and that it's not any given number
                temp_card.type = Type.CHANGECOLOR
                temp_state = self.state_space.get_state(
                    temp_hand, temp_card, temp_card_count
                )
                temp_state0 = self.state_space.get_state(
                    temp_hand, temp_card, temp_card_count0
                )
                temp_state1 = self.state_space.get_state(
                    temp_hand, temp_card, temp_card_count1
                )
                temp_value = self.value.get_value(temp_state)
                temp_value0 = self.value.get_value(temp_state0)
                temp_value1 = self.value.get_value(temp_state1)
                chance_not_playable_card = 1 - (
                    (1 / Color.WILD) + 8 / 108
                )  # 8/108 is for Wilds
                chance_opp_has_card = 1 - (chance_not_playable_card ** card_counts[1])
                next_card_values.append(
                    chance_opp_has_card
                    * temp_value1  # Expected value that opponent had card and played it
                    + (1 - chance_opp_has_card)
                    * (chance_not_playable_card)
                    * temp_value0  # Opponent did not have card but drew and played it
                    + (
                        1 - chance_opp_has_card
                    )  # Opponent did not have it and drew a card they did not play
                    * (1 - chance_not_playable_card)
                    * temp_value
                )
                colors.append(card.color)

        # reward_prime = 0
        if len(valid_actions_idxs) > 0:
            best_value = np.argmax(next_card_values)
            value_color = Color(colors[best_value])
            action_prime = self.hand[valid_actions_idxs[best_value]]
            # Test validity card
            if action_prime is not None:
                assert action_prime in self.hand
                assert action_prime.can_play_on(top_of_pile)
                self.hand.remove(action_prime)
                action_prime = copy.deepcopy(action_prime)
                action_prime.color = value_color

        if self.last_state is not None:
            delta = (
                self.last_reward
                + self.gamma * state_prime_value
                - self.last_state_value
            )

            # Update value net
            if self.update:
                self.value.update(self.last_state, delta)
            self.I *= self.gamma

        self.last_state = state_prime
        self.last_state_value = state_prime_value
        # self.last_reward = reward_prime
        self.last_action = action_prime

        return action_prime

    def on_card_rejection(self, card):
        super().on_card_rejection(card)

    def on_finish(self, winner) -> None:
        # Calc Finish Reward
        win = winner == 0
        reward = 2 * win - 1

        if self.last_state is not None:
            # delta = self.last_reward + self.gamma * reward - self.last_state_value
            delta = reward - self.last_state_value

            # Update value net
            if self.update:
                self.value.update(self.last_state, delta)

        # Reset for next game
        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_reward = 0
        self.last_action = None
        self.value.on_finish()

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save("onestep_rollout")
