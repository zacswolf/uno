from math import exp
from deck import Deck
from player import HumanPlayer
from enums import Direction, Color, Type
from card import Card
import random

class Game(object):
    def __init__(self, num_players, num_bots=0) -> None:
        self.num_players = num_players
        if num_players < 2 or num_players > 6:
            raise Exception("Invalid number of players")
        
        self.deck = Deck()
        

        self.players = []
        for _ in range(num_players):
            self.players.append(HumanPlayer())

        # deal
        for _ in range(7):
            for player in self.players:
                player.get_card(self.draw_card())
        
        # draw top card
        self.pile = [self.draw_card()]
        while(self.pile[-1].type > 9):
            self.pile.append(self.draw_card())

        self.direction = Direction.CLOCKWISE
        self.player_idx = 0

    def draw_card(self) -> Card:
        if (not len(self.deck)):
            # Deck is empty, move pile into deck

            self.deck = random.shuffle(self.pile[:-1])
            self.pile = [self.pile[-1]]

            if (not len(self.deck)):
                # players are holding all of the cards
                raise Exception("Shitty players")
        
        return self.deck.draw_card()


    def start_game(self):
        game_over = False
        try:
            while(not game_over):
                card = self.turn()

                self.pile.append(self.draw_card())

                next_player_idx = (self.player_idx + self.direction) % self.num_players

                if (card is not None):
                    if (card.type == Type.REVERSE):
                        # Reverse
                        self.direction = Direction.CLOCKWISE if (self.direction == Direction.COUNTERCLOCKWISE) else Direction.COUNTERCLOCKWISE
                        next_player_idx = (self.player_idx + self.direction) % self.num_players
                        if (self.num_players == 2):
                            # Reverse 1v1 acts like a skip
                            next_player_idx = (next_player_idx + self.direction) % self.num_players
                    elif (card.type == Type.SKIP):
                        # Skip
                        next_player_idx = (next_player_idx + self.direction) % self.num_players
                    elif (card.type == Type.DRAW2):
                        # Draw 2
                        for _ in range(2):
                            self.players[next_player_idx].get_card(self.draw_card())
                    elif (card.type == Type.DRAW4):
                        # Draw 4
                        for _ in range(4):
                            self.players[next_player_idx].get_card(self.draw_card())
                
                if (len(self.players[self.player_idx].hand) == 0):
                    game_over = True

                    # Notify players for feedback
                    for (idx, player) in enumerate(self.players):
                        player.on_finish((self.player_idx - idx) % self.num_players)
                else:
                    self.player_idx = next_player_idx
        except Exception:
            # Notify players for feedback
            for player in self.players:
                player.on_finish(-1)

    
    def turn(self) -> Card:
        player = self.players[self.player_idx]
        card_counts = list(map(lambda player: len(player.hand), self.players))
        card_counts = card_counts[self.player_idx:] + card_counts[:self.player_idx]
        
        valid_card = False
        card = None
        while (not valid_card):
            card = player.on_turn(self.pile, card_counts)
            if (card is None):
                # Draw
                player.get_card(self.draw_card())
                card = player.on_draw(self.pile, card_counts)
            # If card is none at this point, the player drew a card and didn't play it

            # Check card
            if (card is not None):
                if (card.canPlayOn(self.pile[-1])):
                    valid_card = True

                    # Deal with wild 
                    if(card.color == Color.WILD):
                        color = player.on_choose_wild_color(self.pile, card_counts, card.type)
                        card.color = color
                else:
                    player.on_card_rejection(card)
        
        return card





        
if __name__ == "__main__":
    game = Game(2)
    game.start_game()
        