import flwr as fl
import time
import tensorflow as tf
from multiprocessing import Process
from federated.server.FedAvg import FedAvg
from federated.server.FedAdam import FedAdam
from federated.server.FedYogi import FedYogi
import pickle
from .client import Client

"""
session = tf.compat.v1.Session(graph = tf.Graph() )
with session.graph.as_default():
  tf.keras.backend.set_session(session)
"""


class Federated:
    def __init__(
        self,
        data,
        strategy,
        nbr_clients,
        nbr_rounds,
        directory_name,
        accumulated_data,
        model,
        loss,
        optimizer,
        metrics,
    ):

        self.X_train = data["X_train"]
        self.X_test = data["X_test"]
        self.y_train = data["y_train"]
        self.y_test = data["y_test"]
        self.strategy = strategy
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.directory_name = directory_name
        self.accumulated_data = accumulated_data
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.process = []

    def start_server(
        self,
    ):
        """
        Start a process for the server, call the class associated to the strategy
        """

        print("start server function")

        model = tf.keras.models.clone_model(self.model)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        print("Server model loaded")
        arguments = [
            model,
            self.X_test,
            self.y_test,
            self.nbr_clients,
            self.nbr_rounds,
            self.directory_name,
        ]
        server_process = Process(target=eval(self.strategy), args=arguments)
        server_process.start()
        self.process.append(server_process)

    def start_client(
        self,
        X_train_client,
        y_train_client,
        client_nbr,
    ):
        """
        Start a process for a single client with it associated dataset, dump the results in a pickle
        """

        print("Start client : " + str(client_nbr))
        model = tf.keras.models.clone_model(self.model)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        client = Client(
            model=model,
            X_train=X_train_client,
            y_train=y_train_client,
            X_test=self.X_test,
            y_test=self.y_test,
            client_nbr=client_nbr,
            nbr_rounds=self.nbr_rounds,
            accumulated_data=self.accumulated_data,
        )
        print("client started")
        fl.client.start_numpy_client("[::]:8080", client=client)
        filename = self.directory_name + "/client_number_" + str(client_nbr)
        with open(filename, "wb") as f:
            pickle.dump(client.metrics_list, f)

    def run(self):
        """
        Run the experience, with the server and each clients as a subprocess. The results will be dump in
        a pickle for each one
        """
        self.start_server()
        time.sleep(3)

        # Create partition for each client
        for client in range(self.nbr_clients):
            print(
                "i : " + str(client) + ", size of X_train : " + str(len(self.X_train))
            )
            Client_i = Process(
                target=self.start_client,
                args=(self.X_train[client], self.y_train[client], client),
            )

            Client_i.start()
            self.process.append(Client_i)

        for subprocess in self.process:
            subprocess.join()
