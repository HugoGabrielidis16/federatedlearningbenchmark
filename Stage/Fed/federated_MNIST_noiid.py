#!/usr/bin/python3
import time
from multiprocessing import Process
import pickle

from Fed.Client.client_MNIST_noiid import Client_MNIST_noiid
import flwr as fl

from Fed.Server.server_FedAvg import FedAvg2
from Fed.Server.server_FedAdam import FedAdam2
from Fed.Server.server_FedYogi import FedYogi2
from Fed.Server.server_FedAdagrad import FedAdagrad2


def start_server(strategy, nbr_clients, nbr_rounds, directory_name):
    from Model.model_MNIST import create_model_MNIST
    from data.data_MNIST_noiid.Preprocessing_MNIST_noiid import (Set,X_test,y_test)

    """Start the server with a slightly adjusted FedAvg strategy."""
    model = create_model_MNIST()
    arguments = [model, X_test, y_test, nbr_clients, nbr_rounds, directory_name]
    server = eval(strategy + "2")(*arguments)


def run_MNIST_noiid(strategy, nbr_clients, nbr_rounds,  directory_name, accumulated_data):

    process = []
    # model2 = deepcopy(create_model_JS()) Bug
    server_process = Process(
        target=start_server,
        args=(strategy, nbr_clients, nbr_rounds, directory_name),
    )
    # server_process = Process(target=start_server, args=(nbr_rounds, nbr_clients, 0.2))
    server_process.start()
    process.append(server_process)
    time.sleep(5)

    print("After start")
    for i in range(nbr_clients):
        Client_i = Process(
            target=start_client,
            args=(i,  nbr_clients, directory_name, nbr_rounds, accumulated_data),
        )
        Client_i.start()
        process.append(Client_i)

    for p in process:
        p.join()


def start_client(i, nbr_clients, directory_name, nbr_rounds, accumulated_data):
    from data.data_MNIST_noiid.Preprocessing_MNIST_noiid import (Set,X_test,y_test)
    from Model.model_MNIST import create_model_MNIST


    print("Launching of client" + str(i))
    # Start Flower client
    model = create_model_MNIST()
    client = Client_MNIST_noiid(
        model=model,
        Set = Set[i],
        X_test=X_test,
        y_test=y_test,
        client_nbr=i,
        total_rnd=nbr_rounds,
        accumulated_data = accumulated_data
    )
    fl.client.start_numpy_client("[::]:8080", client=client)
    print("client number " + str(i) + " metrics" + str(client.metrics_list))
    file_name = directory_name + "/client_number_" + str(i)
    with open(file_name, "wb") as f:
        pickle.dump(client.metrics_list, f)
