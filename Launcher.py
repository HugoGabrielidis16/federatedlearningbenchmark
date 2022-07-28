import os
import argparse
import datetime
import time
import tensorflow as tf
import warnings
from multiprocessing import Process


from centralized import Centralized
from federated.federated import Federated
from model.model import FLModel
from data.data import DataFactory
from results.load_result import create_curves


warnings.filterwarnings("ignore")


"""
Launcher file that will launch both experiment : Centraized
and Federated.
Creating the results curves at the end of both experiments.
"""


def define_parser():
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument("--nbr_clients", type=int, required=True)
    parser.add_argument("--nbr_rounds", type=int, required=True)
    parser.add_argument(
        "--Dataset",
        type=str,
        required=True,
        help="Choose a dataset between JS, CIC_IDS2017, MovieLens, CIFAR10, Shakespeare, MNIST, DisasterTweets, IMDB, Bostonhouse.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Choose an aggregation strategy between : FedAvg, FedAdam, FedYogi or FedAdagrad",
        required=True,
    )
    parser.add_argument("--accumulated_data", type=str)
    parser.add_argument("--centralized_percentage", type=float)
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


def load_model(name):
    """
    Load the model and the diffrents parameters needed to compile it < use that since there are issue
    cloning already compiled model using keras >
    """
    model_ = FLModel(name)
    model = model_.model
    loss = model_.loss
    optimizer = model_.optimizer
    metrics = model_.metrics

    return model, loss, optimizer, metrics


def main() -> None:
    args = define_parser()
    directory_name = create_directory(args=args)
    model, loss, optimizer, metrics = load_model(args.Dataset)

    dataset = DataFactory().load_data(args.Dataset, args.nbr_clients, args.nbr_rounds)
    dataset_name = args.Dataset

    print(
        "-------------------" * 5 + " Start of Centralized " + "-------------------" * 5
    )
    start_centralized = time.time()
    centralized_run = Centralized(
        dataset=dataset,
        directory_name=directory_name,
        nbr_rounds=args.nbr_rounds,
        nbr_clients=args.nbr_clients,
        accumulated_data=eval(args.accumulated_data),
        percentage=args.centralized_percentage,
        model=model,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    """ centralized_run.run() """
    end_centralized = time.time()
    print()
    print(f"Runtime of centralized is {end_centralized - start_centralized}")
    print()
    print("-------------------" * 5 + "Start of Federated" + "-----------------" * 5)
    federated_run = Federated(
        data=dataset,
        directory_name=directory_name,
        nbr_rounds=args.nbr_rounds,
        nbr_clients=args.nbr_clients,
        strategy=args.strategy,
        accumulated_data=eval(args.accumulated_data),
        model=model,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
    start_federated = time.time()
    federated_run.run()
    end_federated = time.time()
    print()
    print(f"Runtime of federated is {end_federated - start_federated}")
    create_curves(experience_path=directory_name)


if __name__ == "__main__":
    main()
