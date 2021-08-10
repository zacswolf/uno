from enums import Color
from card import Card

class Player(object):
    def __init__(self) -> None:
        self.hand = []

    def get_card(self, card):
        self.hand.append(card)

    # Return Card to play card or None to draw
    def on_turn(self, pile, card_counts) -> Card:
        raise NotImplementedError()
    
    # Return Card to play card or None to skip
    def on_draw(self, pile, card_counts):
        raise NotImplementedError()

    def on_choose_wild_color(self, pile, card_counts, type) -> Color:
        raise NotImplementedError()

    # Game rejects player's card
    def on_card_rejection(self, pile, card_counts, card) -> None:
        self.hand.append(card)

    # Winner is a player index relative to player or -1 if no-one wins
    def on_finish(self, winner) -> None:
        return
    
class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def on_turn(self, pile, card_counts):
        print("CARD COUNTS:\t", card_counts)
        print("PILE: NEW\t", list(reversed(pile[-len(card_counts):])) if (len(pile) >= len(card_counts)) else pile, "\tOLD")
        
        # default sort
        self.hand.sort(key = lambda c: c.type)
        self.hand.sort(key = lambda c: c.color)

        print("HAND:", ["%i: %s" % (i, c) for (i, c) in enumerate(self.hand)])

        print("What do you wanna play?")
        while (True):
            play = input()
            if (play == 'd' or play == ""):
                return None
            else:
                card = None
                try:
                    play = int(play)
                    card = self.hand.pop(play)
                    return card
                except ValueError: 
                    print("Please give a valid input")
                    continue 
                except IndexError: 
                    print("Please give a valid input")
                    continue 
    
    def on_draw(self, pile, card_counts):
        return self.on_turn(pile, card_counts)

    def on_choose_wild_color(self, pile, card_counts, type):
        print("What color do you want to change it to?")
        while (True):
            wild_color = input()
            if (not len(wild_color)):
                print("Please give a valid input")
                continue
            wild_color = wild_color[0]
            if wild_color == 'r':
                wild_color = Color.RED
            elif wild_color == 'b':
                wild_color = Color.BLUE
            elif wild_color == 'g':
                wild_color = Color.GREEN
            elif wild_color == 'y':
                wild_color = Color.YELLOW
            else:
                print("Please give a valid input")
                continue
            return wild_color
            
    def on_card_rejection(self, pile, card_counts, card):
        super().on_card_rejection(pile, card_counts, card)
        print("Card rejected")
