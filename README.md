# The game of Uno

Run `python src/game.py --num_players 2 --players [player_type] [player_type]` to run the game

Args list:
```
--num_players **int** | Number of players
--draw_skip **boolean** | Don't skip a players turn if they have to draw 2/4 
--num_cards **int** | Initial number of cards per player
--players **string** | List name of player types (string count must match number of players)
--num_games **int** | Number of games to play
--value_net **string** | File path of value net to initialize with
--policy_net **string** | File path of policy net to initialize with
--update **boolean** | Update the RL model weights as they play
--alternate **boolean** | Alternate which player starts first
--conf **string** | Config file to use
```

Player type list:
```
human
draw
random
noob
basic
decent
decent2
decent3
decent4
firstrlplayer
secrlplayer
reinvalact
reinvalactsoft
reinvalactsoft2
onestepac
onestepacsoft
onesteprollout
qlearn
qlearnbatch
sarsa
```

Args can also be controlled in config files and called via: `python src/game.py --conf example.yaml`

To add a new bot see the str_to_player function in src/player.py

Must use a python version >= 3.10
