import os
import time
from multiprocessing import Process
import pickle
from copy import deepcopy
from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017

from Fed.Client.client_CIC_IDS2017 import Client_CIC_IDS2017
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2
import tensorflow as tf
import numpy as np


def start_server(
    strategy,
    nbr_clients,
    nbr_rounds,
    X_test_centralized,
    y_test_centralized,
    directory_name,
):

    from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017
     
    """Start the server with a slightly adjusted FedAvg strategy."""
    model = create_model_CIC_IDS2017()
    arguments = [
        model,
        X_test_centralized,
        y_test_centralized,
        nbr_clients,
        nbr_rounds,
        directory_name,
    ]
    server = eval(strategy + "2")(*arguments)


def run_CIC_IDS2017(strategy, nbr_clients, nbr_rounds, timed, directory_name):

    from data.data_CIC_IDS2017.experiment import (
        Set,
        X_test_centralized,
        y_test_centralized,
    )  # Did it this way in order to not load the entire dataset for each client
    print(type(X_test_centralized))

    X_test_centralized = np.reshape(X_test_centralized,(X_test_centralized[0], 1, X_test_centralized[1]))

    
    process = []
    server_process = Process(
        target=start_server,
        args=(
            strategy,
            nbr_clients,
            nbr_rounds,
            X_test_centralized,
            y_test_centralized,
            directory_name,
        ),
    )
    server_process.start()
    process.append(server_process)
    time.sleep(5)

    for i in range(nbr_clients):

        Client_i = Process(
            target=start_client,
            args=(
                i,
                timed,
                Set,
                X_test_centralized,
                y_test_centralized,
                directory_name,
                nbr_rounds,
            ),
        )
        Client_i.start()
        process.append(Client_i)

    for p in process:
        p.join()


def start_client(
    i,
    timed,
    Set,
    X_test_centralized,
    y_test_centralized,
    directory_name,
    nbr_rounds,
):
    from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017

    model = create_model_CIC_IDS2017()
    client = Client_CIC_IDS2017(
        model=model,
        Set=Set[i],  # Set for the i-th client
        X_test=X_test_centralized,
        y_test=y_test_centralized,
        client_nbr=i,
        timed=timed,
        total_rnd=nbr_rounds,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
    print("client number " + str(i) + " metrics" + str(client.metrics_list))
    file_name = directory_name + "/client_number_" + str(i)
    with open(file_name, "wb") as f:
        pickle.dump(client.metrics_list, f)
