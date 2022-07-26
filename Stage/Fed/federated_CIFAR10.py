#!/usr/bin/python3
import time
from multiprocessing import Process
import pickle

from Fed.Client.client import Client_Test
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2


def start_server(strategy, nbr_clients, nbr_rounds, directory_name):
    from Model.model_CIFAR10 import create_model_CIFAR10
    from data.data_CIFAR10.Preprocessing_CIFAR10 import X_test, y_test

    """Start the server with a slightly adjusted FedAvg strategy."""
    model = create_model_CIFAR10()
    arguments = [model, X_test, y_test, nbr_clients, nbr_rounds, directory_name]
    print("Before eval")
    server = eval(strategy + "2")(*arguments)


def run_CIFAR10(strategy, nbr_clients, nbr_rounds, directory_name, accumulated_data):

    process = []
    # model2 = deepcopy(create_model_JS()) Bug
    server_process = Process(
        target=start_server,
        args=(strategy, nbr_clients, nbr_rounds, directory_name),
    )
    # server_process = Process(target=start_server, args=(nbr_rounds, nbr_clients, 0.2))
    server_process.start()
    process.append(server_process)
    time.sleep(10)

    print("After start")
    print(nbr_clients)
    for i in range(nbr_clients):
        Client_i = Process(
            target=start_client,
            args=(i,nbr_clients,nbr_rounds,directory_name, accumulated_data,),
        )
        Client_i.start()
        process.append(Client_i)

    for p in process:
        p.join()


def start_client(i,nbr_clients ,nbr_rounds,directory_name, accumulated_data):
    print("Starting clinet")
    from data.data_CIFAR10.Preprocessing_CIFAR10 import (
        X_test,
        X_train,
        y_test,
        y_train,
    )
    from Model.model_CIFAR10 import create_model_CIFAR10
    X_train_i = X_train[
        int((i / nbr_clients) * len(X_train)) : int(
            ((i + 1) / nbr_clients) * len(X_train)
        )
    ]
    y_train_i = y_train[
        int((i / nbr_clients) * len(y_train)) : int(
            ((i + 1) / nbr_clients) * len(y_train)
        )
    ]  # So each client have a different dataset to train on

    print("Launching of client" + str(i))
    # Start Flower client
    model = create_model_CIFAR10()
    client = Client_Test(
        model=model,
        X_train=X_train_i,
        y_train=y_train_i,
        X_test=X_test,
        y_test=y_test,
        client_nbr=i,
        total_rnd=nbr_rounds,
        accumulated_data = accumulated_data,
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
    print("client number " + str(i) + " metrics" + str(client.metrics_list))
    file_name = directory_name + "/client_number_" + str(i)

    for i in range(len(client.duration) - 1):
        client.duration[i + 1] += client.duration[i]
    
    with open(file_name, "wb") as f:
        pickle.dump(client.metrics_list, f)
        pickle.dump(client.duration,f) 

