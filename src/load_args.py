import argparse
import os
from datetime import datetime
from dataclasses import dataclass


# Nested Argument Data Classes
@dataclass
class ArgsGameShared:
    run_name: str
    model_dir: str
    num_players: int = 2
    num_games: int = 1
    update: bool = True


@dataclass
class ArgsGamePrivate:
    root_file: str
    config_dir: str
    num_cards: int = 7
    with_replacement: bool = True
    alternate: bool = False
    draw_skip: bool = True


@dataclass
class ArgsGame:
    shared: ArgsGameShared
    private: ArgsGamePrivate


@dataclass
class ArgsPlayer:
    player: str
    value_net: str = ""
    policy_net: str = ""
    gamma: float = 1.0
    epsilon: float = 1.0
    player_idx: int = -1  # defined in code


@dataclass
class Args:
    game: ArgsGame
    players: list[ArgsPlayer]


def load_args() -> Args:
    """Loads in the argument in decreasing priority:
    1. Argument
    2. Config file
    3. Data Class defaults

    Returns:
        Args: Arguments
    """

    my_parser = argparse.ArgumentParser(description="Uno game")
    # Arguments, should have None as default
    my_parser.add_argument(
        "-n",
        "--num_players",
        type=int,
        choices=range(2, 11),
        metavar="[2-10]",
        help="Number of players",
    )
    my_parser.add_argument(
        "--draw_skip",
        action=argparse.BooleanOptionalAction,
        help="Don't skip a players turn if they have to draw 2/4",
    )
    my_parser.add_argument(
        "--with_replacement",
        action=argparse.BooleanOptionalAction,
        help="Deck is drawn with replacement",
    )
    my_parser.add_argument(
        "--num_cards",
        type=int,
        help="Initial number of cards per player",
    )
    my_parser.add_argument(
        "--players",
        nargs="+",
        help="List of players using player strings, required if no config",
    )
    my_parser.add_argument(
        "--num_games",
        type=int,
        help="Number of games to play",
    )
    # my_parser.add_argument(
    #     "--value_net",
    #     nargs="+",
    #     help="File locations of value_net to initialize with",
    # )
    # my_parser.add_argument(
    #     "--policy_net",
    #     nargs="+",
    #     help="File locations of policy_net to initialize with",
    # )
    # my_parser.add_argument(
    #     "--gamma",
    #     nargs="+",
    #     help="Gamma for each player",
    # )
    my_parser.add_argument(
        "--update",
        action=argparse.BooleanOptionalAction,
        help="Don't update any of the rl bots, just evaluate",
    )
    my_parser.add_argument(
        "--alternate",
        action=argparse.BooleanOptionalAction,
        help="Alternate which player starts first",
    )
    my_parser.add_argument(
        "--conf",
        type=str,
        help="Config file to use",
    )

    args = my_parser.parse_args()

    # assert args.num_players == 2

    root_file = os.path.dirname(__file__)
    config_dir = os.path.join(root_file, "../configs/")

    arg_dict = dict()
    arg_dict["game"] = {"shared": dict(), "private": dict()}
    arg_dict["game"]["shared"] = {
        "num_players": args.num_players,
        "num_games": args.num_games,
        "update": args.update,
        "run_name": datetime.now().strftime("%m_%d_%H_%M_%S"),
        "model_dir": os.path.join(root_file, "../models/"),
    }
    arg_dict["game"]["private"] = {
        "with_replacement": args.with_replacement,
        "num_cards": args.num_cards,
        "alternate": args.alternate,
        "draw_skip": args.draw_skip,
        "root_file": root_file,
        "config_dir": config_dir,
    }

    # No longer support player specific args via commandline
    arg_dict["players"] = []
    # if args.players:
    # value_nets = args.value_net
    # policy_nets = args.policy_net
    # gammas = args.gamma

    # # Process value_net
    # if value_nets:
    #     assert len(value_nets) <= len(args.players)
    #     if len(value_nets) < len(args.players):
    #         # Pad value net arg
    #         value_nets += [""] * (len(args.players) - len(value_nets))
    # else:
    #     value_nets = [""] * len(args.players)

    # # Process policy_net
    # if policy_nets:
    #     assert len(policy_nets) <= len(args.players)
    #     if len(policy_nets) < len(args.players):
    #         # Pad policy net arg
    #         policy_nets += [""] * (len(args.players) - len(policy_nets))
    # else:
    #     policy_nets = [""] * len(args.players)

    # # Process gammas
    # if gammas:
    #     assert len(gammas) <= len(args.players)
    #     if len(gammas) < len(args.players):
    #         # Pad policy net arg
    #         gammas += [1] * (len(args.players) - len(gammas))
    # else:
    #     gammas = [1] * len(args.players)

    # Add to arg_dict

    # for (player, value_net, policy_net, gamma) in zip(
    #     args.players, value_nets, policy_nets, gammas, strict=True
    # ):
    #     arg_dict["players"].append(
    #         {
    #             "player": player,
    #             "value_net": value_net,
    #             "policy_net": policy_net,
    #             "gamma": gamma,
    #         }
    #     )

    if args.players:
        for player in args.players:
            arg_dict["players"].append({"player": player})

    if args.conf:
        # we have a config file
        import yaml

        with open(os.path.join(config_dir, args.conf), "r") as file:
            config_dict = yaml.safe_load(file)

            # arg_dict["game"]["private"].update(config_dict["game"]["private"])
            config_dict["game"]["private"].update(
                (k, v) for k, v in arg_dict["game"]["private"].items() if v is not None
            )
            arg_dict["game"]["private"] = config_dict["game"]["private"]

            # arg_dict["game"]["shared"].update(config_dict["game"]["shared"])
            config_dict["game"]["shared"].update(
                (k, v) for k, v in arg_dict["game"]["shared"].items() if v is not None
            )
            arg_dict["game"]["shared"] = config_dict["game"]["shared"]

            players_list = None
            if arg_dict["players"]:
                players_list = arg_dict["players"]
            elif "players" in config_dict and config_dict["players"]:
                players_list = config_dict["players"]
            else:
                assert False

            arg_dict["players"] = players_list

    # Create sub args data classes

    # arg_dict["game"]["private"] = ArgsGamePrivate(**arg_dict["game"]["private"])
    arg_dict["game"]["private"] = ArgsGamePrivate(
        **{k: v for k, v in arg_dict["game"]["private"].items() if v is not None}
    )

    # arg_dict["game"]["shared"] = ArgsGameShared(**arg_dict["game"]["shared"])
    arg_dict["game"]["shared"] = ArgsGameShared(
        **{k: v for k, v in arg_dict["game"]["shared"].items() if v is not None}
    )

    arg_dict["game"] = ArgsGame(**arg_dict["game"])

    # Create players data class
    for idx, player in enumerate(arg_dict["players"]):
        arg_dict["players"][idx] = ArgsPlayer(**player)

    ab = Args(**arg_dict)

    return ab
