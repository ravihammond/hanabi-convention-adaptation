import argparse
import os
import sys
import numpy as np
import re

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
from model_zoo import model_zoo

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass


def evaluate_model(args):
    if args.output is not None:
        sys.stdout = Logger(args.output)

    weight_files = load_weights(args)
    # regex_search = re.search("([0-9]+)", args.weight1)
    # print(f"epoch: {regex_search.group(1)}")
    scores, actors = run_evaluation(args, weight_files)

    print_scores(scores)
    print_actor_stats(actors, 0)
    print_actor_stats(actors, 1)
    print("==========")

    if args.output is not None:
        sys.stdout.logfile.close()
        sys.stdout = sys.stdout.terminal


def load_weights(args):
    weight_files = []
    if args.num_player == 2:
        if args.weight2 is None:
            args.weight2 = args.weight1
        weight_files = [args.weight1, args.weight2]
    elif args.num_player == 3:
        if args.weight2 is None:
            weight_files = [args.weight1 for _ in range(args.num_player)]
        else:
            weight_files = [args.weight1, args.weight2, args.weight3]

    for i, wf in enumerate(weight_files):
        if wf in model_zoo:
            weight_files[i] = model_zoo[wf]

    assert len(weight_files) == 2
    return weight_files


def run_evaluation(args, weight_files):
    _, _, _, scores, actors = evaluate_saved_model(
        weight_files,
        args.num_game,
        args.seed,
        args.bomb,
        num_run=args.num_run,
        convention=args.convention,
        convention_sender=args.convention_sender,
        override=[args.override0, args.override1],
    )

    return scores, actors


def print_scores(scores):
    non_zero_scores = [s for s in scores if s > 0]
    print(f"non_zero_mean: %.3f" % (
        0 if len(non_zero_scores) == 0 else np.mean(non_zero_scores)))
    print(f"bomb_out_rate: {100 * (1 - len(non_zero_scores) / len(scores)):.2f}%")


def print_actor_stats(actors, player):
    print_played_card_knowledge(actors, player)
    print_move_stats(actors, player)
    print_convention_stats(actors, player)


def print_played_card_knowledge(actors, player):
    card_stats = []
    for i, g in enumerate(actors):
        if i % 2 == player:
            card_stats.append(g.get_played_card_info())
    card_stats = np.array(card_stats).sum(0)
    total_played = sum(card_stats)

    print(f"actor{player}_total_cards_played: {total_played}")
    for i, ck in enumerate(["none", "color", "rank", "both"]):
        percentage = (card_stats[i] / total_played) * 100
        percentage = percent(card_stats[i], total_played)
        print(f"actor{player}_card_played_knowledge_{ck}: {card_stats[i]}",
              f"({percentage:.1f}%)")


def print_move_stats(actors, player):
    colour_move_map = ["red", "yellow", "green", "white", "blue"]
    rank_move_map = ["1", "2", "3", "4", "5"]

    print_move_type_stat(actors, player, "play")
    print_move_type_stat(actors, player, "discard")
    print_move_type_stat(actors, player, "hint_colour")
    print_move_type_stats(actors, player, "hint", colour_move_map, "hint_colour")
    print_move_type_stat(actors, player, "hint_rank")
    print_move_type_stats(actors, player, "hint", rank_move_map, "hint_rank")


def print_move_type_stats(actors, player, move_type, move_map, move_total):
    total = sum_stats(move_total, actors, player)
    for move in move_map:
        move_total = sum_stats(move_type + "_" + move, actors, player)
        print(f"actor{player}_{move_type}_{move}: {move_total}", 
                f"({percent(move_total, total):.1f}%)")


def print_convention_stats(actors, player):
    available = sum_stats("convention_available", actors, player)
    played = sum_stats("convention_played", actors, player)
    played_correct = sum_stats("convention_played_correct", actors, player)
    played_incorrect = sum_stats("convention_played_incorrect", actors, player)
    played_correct_available_percent = percent(played_correct, available)
    played_correct_played_percent = percent(played_correct, played)

    print(f"actor{player}_convention_available: {int(available)}")
    print(f"actor{player}_convention_played: {played}")
    print(f"actor{player}_convention_played_correct: {played_correct}", 
            f"({played_correct_available_percent:.1f}%",
            f"{played_correct_played_percent:.1f}%)")
    print(f"actor{player}_convention_played_incorrect: {played_incorrect}")

    for i in range(5):
        playable = sum_stats(f"convention_played_{i}_playable", actors, player)
        print(f"actor{player}_convention_played_{i}_playable: {playable}")


def print_move_type_stat(actors, player, move_type):
    count = sum_stats(move_type, actors, player)
    print(f"actor{player}_{move_type}: {int(count)}")


def sum_stats(key, actors, player):
    stats = []
    for i, g in enumerate(actors): 
        if i % 2 == player:
            if key in g.get_stats():
                stats.append(g.get_stats()[key])
    return int(sum(stats))

def percent(n, total):
    if total == 0:
        return 0
    return (n / total) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight1", default=None, type=str, required=True)
    parser.add_argument("--weight2", default=None, type=str)
    parser.add_argument("--weight3", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--num_player", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bomb", default=0, type=int)
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument(
        "--num_run",
        default=1,
        type=int,
        help="num of {num_game} you want to run, i.e. num_run=2 means 2*num_game",
    )
    parser.add_argument("--convention", default="None", type=str)
    parser.add_argument("--convention_sender", default=0, type=int)
    parser.add_argument('--override0', action='store_true')
    parser.add_argument('--override1', action='store_true')
    parser.set_defaults(override0=False, override1=False)
    args = parser.parse_args()

    print("override0")
    print(args.override0)
    print("override1")
    print(args.override1)

    evaluate_model(args)

