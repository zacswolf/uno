import copy
import logging
from player import Player
from players.common import action_space, state_space, policy_nets, value_nets

class OneStepActorCritic(Player):
    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.policy = policy_nets.PolNetValActions(
            self.action_space, self.state_space.size(), player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)
        self.gamma = player_args.gamma

        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_reward = 0
        self.last_action = None

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state prime
        state_prime = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state_prime.shape[0] == self.state_space.size()
        state_prime_value = self.value.get_value(state_prime)
        logging.debug(f"state prime value: {state_prime_value}")
        
        
        
        # Get card
        action_prime = self.policy.get_action(self.hand, state_prime, top_of_pile)
        # reward_prime = 0

        # Test validity card
        if action_prime is not None:
            assert action_prime in self.hand
            assert action_prime.can_play_on(top_of_pile)

        

        if self.last_state is not None:
            delta = self.last_reward + self.gamma * state_prime_value - self.last_state_value

            # Update value net
            self.value.update(self.last_state, delta)

            # Update policy net
            self.policy.update(self.last_state, self.last_action, self.I * delta)

            self.I *= self.gamma

        self.last_state = state_prime
        self.last_state_value = state_prime_value
        # self.last_reward = reward_prime 
        self.last_action = action_prime
        
        # Return card
        if action_prime:
            self.hand.remove(action_prime)
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
            self.value.update(self.last_state, delta)

            # Update policy net
            self.policy.update(self.last_state, self.last_action, delta)

        # Reset for next game
        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_reward = 0
        self.last_action = None

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save()
            self.policy.save()



class OneStepActorCriticSoft(Player):
    def __init__(self, player_args, game_args) -> None:
        super().__init__(player_args, game_args)

        self.num_games = game_args.num_games
        self.game_num = 0

        self.gamma = player_args.gamma

        self.update = game_args.update

        self.state_space = state_space.SSRep1(game_args)
        self.action_space = action_space.ASRep1(game_args)

        self.policy = policy_nets.PolNetValActionsSoftmax(
            self.action_space, self.state_space.size(), player_args, game_args
        )
        self.value = value_nets.ValueNet1(self.state_space, player_args, game_args)

        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_reward = 0
        self.last_action = None

    def on_turn(self, pile, card_counts, drawn):
        top_of_pile = pile[-1]

        # Get state prime
        state_prime = self.state_space.get_state(self.hand, top_of_pile, card_counts)
        assert state_prime.shape[0] == self.state_space.size()
        state_prime_value = self.value.get_value(state_prime)
        logging.debug(f"state prime value: {state_prime_value}")
        
        
        
        # Get card
        action_prime = self.policy.get_action(self.hand, state_prime, top_of_pile)
        # reward_prime = 0

        # Test validity card
        if action_prime is not None:
            assert action_prime in self.hand
            assert action_prime.can_play_on(top_of_pile)

        

        if self.last_state is not None:
          
            delta = self.last_reward + self.gamma * state_prime_value - self.last_state_value

            # Update value net
            self.value.update(self.last_state["state"], delta)

            # Update policy net
            self.policy.update(self.last_state, self.last_action, self.I * delta)

            self.I *= self.gamma

        wrapped_state = {
            "state": state_prime,
            "hand": copy.deepcopy(self.hand),
            "top_of_pile": copy.copy(top_of_pile),
        }
        self.last_state = wrapped_state
        self.last_state_value = state_prime_value
        # self.last_reward = reward_prime 
        self.last_action = action_prime
        
        # Return card
        if action_prime:
            self.hand.remove(action_prime)
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
            self.value.update(self.last_state["state"], delta)

            # Update policy net
            self.policy.update(self.last_state, self.last_action, delta)

        # Reset for next game
        self.I = 1
        self.last_state = None
        self.last_state_value = 0
        self.last_reward = 0
        self.last_action = None

        self.game_num += 1

        if self.game_num == self.num_games:
            # Save models
            self.value.save()
            self.policy.save()