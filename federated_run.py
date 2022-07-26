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
import FLconfig


tf.disable_v2_behavior()


from results.load_result import create_curves




def take_metrics(dataset):
    if dataset in ["MNIST", "CIFAR10"]:
        return "sparse_categorical_accuracy"
    elif dataset in ["IMDB","DisasterTweets"]:
        return "binary_accuracy"

    elif dataset in ["JS","Bostonhouse"]:
        return "mean_squared_error"


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument(
        "--nbr_clients",
        type=int,
        choices=range(1, 101),
        required=True,
    )

    parser.add_argument(
        "--directory_name",
        type=str,
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
    
    args = parser.parse_args()
    directory_name = args.directory_name
    try :
        os.mkdir(directory_name)
    except:
        pass

    metrics = take_metrics(args.Dataset)
    dataset_name = args.Dataset
    graph = tf.get_default_graph()
    print("-------------------" * 5 + "Start of Federated" + "-----------------" *5)
    federated_run = Federated(
            dataset_name = dataset_name,
            directory_name = directory_name, 
            nbr_rounds = args.nbr_rounds, 
            nbr_clients = args.nbr_clients, 
            strategy = args.strategy,
            accumulated_data = eval(args.accumulated_data),
            graph = graph
            ) 
    start_federated = time.time()
    federated_run.run()
    end_federated = time.time()
    print()
    print(f"Runtime of federated is {end_federated - start_federated}")
    create_curves(experience_path=directory_name)

if __name__ == "__main__":
    main()
