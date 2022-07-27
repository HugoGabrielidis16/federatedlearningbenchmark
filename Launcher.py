from multiprocessing import Process
import os
import sys
import argparse
import traceback
import signal
import datetime
import time
from copy import deepcopy
import tensorflow.compat.v1 as tf
import warnings

warnings.filterwarnings("ignore")

from centralized.centralized import Centralized
from federated.federated import Federated
from model.model import FLModel
from data.data import Data
import FLconfig


tf.disable_v2_behavior()


from results.load_result import create_curves


def main() -> None:
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
    directory_name = (
        "results/"
        + args.Dataset
        + "_"
        + args.strategy
        + "_clients_"
        + str(args.nbr_clients)
        + "_rounds_"
        + str(args.nbr_rounds)
        + "_"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    try:
        os.mkdir(directory_name)
    except:
        pass

    model_ = FLModel(args.Dataset)
    loss = model_.loss
    optimizer = model_.optimizer
    metrics_ = model_.metrics

    model_centralized = tf.keras.models.clone_model(model_)
    model_centralized.compile(loss=loss, optimizer=optimizer, metrics=metrics_)

    model_federated = tf.keras.models.clone_model(model_)

    dataset = Data(args.Dataset)
    metrics = take_metrics(args.Dataset)
    dataset_name = args.Dataset
    graph = tf.get_default_graph()
    # """
    print(
        "-------------------" * 5 + " Start of Centralized " + "-------------------" * 5
    )
    start_centralized = time.time()
    centralized_run = Centralized(
        model=model_centralized,
        data=dataset,
        directory_name=directory_name,
        nbr_rounds=args.nbr_rounds,
        nbr_clients=args.nbr_clients,
        metrics=metrics,
        accumulated_data=eval(args.accumulated_data),
        percentage=args.centralized_percentage,
        graph=graph,
    )

    centralized_run.run()
    end_centralized = time.time()
    print
    print(f"Runtime of centralized is {end_centralized - start_centralized}")
    # """
    print()
    print("-------------------" * 5 + "Start of Federated" + "-----------------" * 5)
    federated_run = Federated(
        dataset_name=dataset_name,
        data=dataset,
        directory_name=directory_name,
        nbr_rounds=args.nbr_rounds,
        nbr_clients=args.nbr_clients,
        strategy=args.strategy,
        accumulated_data=eval(args.accumulated_data),
        loss=loss,
        optimizer=optimizer,
        metrics=metrics_,
        graph=graph,
    )
    start_federated = time.time()
    federated_run.run()
    end_federated = time.time()
    print()
    print(f"Runtime of federated is {end_federated - start_federated}")
    create_curves(experience_path=directory_name)


if __name__ == "__main__":
    main()
