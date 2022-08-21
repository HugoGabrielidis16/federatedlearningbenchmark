from multiprocessing import Process
import os
from pickle import FALSE
import sys
import argparse


PATH = os.getcwd()
""" sys.path.append(PATH + "/Centralized")
import centralized_JS
import centralized_CIC_IDS2017
import centralized_MovieLens

import centralized_Shakespeare """
from Centralized.centralized_JS import run_centralized_JS
from Centralized.centralized_CIFAR10 import run_centralized_CIFAR10
from Centralized.centralized_MNIST import run_centralized_MNIST
from Centralized.centralized_CIC_IDS2017 import run_centralized_CIC_IDS2017
from Centralized.centralized_Shakespeare import run_centralized_Shakespeare
from Centralized.centralized_DisasterTweets import run_centralized_DisasterTweets
from Centralized.centralized_IMDB import run_centralized_IMDB
from Centralized.centralized_Bostonhouse import run_centralized_Bostonhouse

from Centralized.centralized_MNIST_noiid import run_centralized_MNIST_noiid


from Fed.federated_JS import run_JS
from Fed.federated_CIFAR10 import run_CIFAR10
from Fed.federated_MNIST import run_MNIST
from Fed.federated_CIC_IDS2017 import run_CIC_IDS2017
from Fed.federated_DisasterTweets import run_DisasterTweets
from Fed.federated_IMDB import run_IMDB
from Fed.federated_Bostonhouse import run_Bostonhouse
from Fed.federated_MNIST_noiid import run_MNIST_noiid
import traceback
import signal
import datetime

import FLconfig

import time


from results.server_centralized import create_curves
from results.only_clients import create_curves_clients
actual_time = time.ctime()
actual_time = actual_time.split(" ")


def __signal_code_to_name(code):
    for s in signal.Signals:
        if s.value == code:
            return s.name
    return "Unknown signal"


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!", __signal_code_to_name(sig))
    print(traceback.format_exc())
    sys.exit(0)


def main() -> None:
    """if threading.current_thread() == threading.main_thread():
    for sig in signal.Signals:
        try:
            if sig.name == "SIGCHLD":
                continue
            signal.signal(sig, signal_handler)
        except OSError:
            print(("Skipping signal", sig))"""

    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument(
        "--nbr_clients",
        type=int,
        choices=range(1, 101),
        required=True,
    )
    parser.add_argument(
        "--nbr_rounds",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--Dataset",
        type=str,
        choices=[
            "JS",  
            "CIC_IDS2017",  
            "MovieLens",
            "CIFAR10",
            "Shakespeare",
            "MNIST",
            "DisasterTweets",
            "IMDB",
            "Bostonhouse",
            "MNIST_noiid"
        ],
        required=True,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[
            "FedAvg",
            "FedAdam",
            "FedAdagrad",
            "FedYogi",
            "AggregateCustomMetricStrategy",
        ],
        required=True,
    )
    parser.add_argument(
      "--accumulated_data",
      type = str
    )
    parser.add_argument(
      "--centralized_percentage",
      type = float
    )
    args = parser.parse_args()
    directory_name = (
        "results/25_07_2022/"
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
    centralized = "run_centralized_" + args.Dataset

    federated = "run_" + args.Dataset
    arguments = [
        args.strategy,
        args.nbr_clients,
        args.nbr_rounds,
        directory_name,
        eval(args.accumulated_data),
    ]
    #""" 
    print("-------------------" * 4 + "Start of Centralized" + "-----------------" * 4)
    start_centralized = time.time()
    centralized_process = Process(
        target=eval(centralized),
        args=(args.nbr_rounds, args.nbr_clients, directory_name,eval(args.accumulated_data) ,args.centralized_percentage,),
    )
    centralized_process.start()
    centralized_process.join()
    end_centralized = time.time()
    print(f"Runtime of centralized is {end_centralized - start_centralized}")
    #"""
    #"""
    print("-------------------" * 4 + "Start of Federated" + "-----------------" * 4)

    start_federated = time.time()
    eval(federated)(*arguments)
    end_federated = time.time()
    print(f"Runtime of federated is {end_federated - start_federated}")
    create_curves(experience_path=directory_name)
    create_curves_clients(experience_path=directory_name)
    #"""
if __name__ == "__main__":
    main()
