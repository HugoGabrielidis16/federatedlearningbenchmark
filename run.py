import os
import argparse
import datetime
import tensorflow as tf
import warnings
from multiprocessing import Process


from centralized_run import run_centralized
from federated_run import run_federated
from federated.federated import Federated


warnings.filterwarnings("ignore")


"""
Launcher file that will launch both experiment : Centralized
and Federated.
Creating the results curves at the end of both experiments.
"""


def define_parser():
    """
    Create the parser for the argument needed :
        - Dataset
        - strategy
        - nbr_rounds
        - nbr_clients
        - accumulated_data
        - centralized_percentage

    """

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--Dataset",
        type=str,
        help="Choose a dataset between JS, CIC_IDS2017, MovieLens, CIFAR10, Shakespeare, MNIST, DisasterTweets, IMDB, Bostonhouse.",
        default="MNIST",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Choose an aggregation strategy between : FedAvg, FedAdam, FedYogi or FedAdagrad",
        default="FedAvg",
    )
    parser.add_argument("--nbr_clients", type=int, default=2)
    parser.add_argument("--nbr_rounds", type=int, default=2)

    parser.add_argument("--accumulated_data", type=str, default="False")
    parser.add_argument("--centralized_percentage", type=float, default=1)
    args = parser.parse_args()
    return args


def create_directory(args):
    """
    Create the directory
    """
    directory_name = f"results/{args.Dataset}/{args.strategy}/{args.nbr_clients}_clients{args.nbr_rounds}_rounds_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        os.mkdir(directory_name)
    except:
        pass
    return directory_name


if __name__ == "__main__":
    args = define_parser()
    directory_name = create_directory(args=args)
    run_centralized(args, directory_name=directory_name)
    run_federated(args, directory_name)
