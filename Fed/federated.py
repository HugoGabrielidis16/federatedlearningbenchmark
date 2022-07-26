from datetime import date
from multiprocessing import Process
from multiprocessing.connection import Client
import flwr as fl
from Fed.Client.client import Client_Test


class federated:
    def __init__(
        self,
        model,
        strategy,
        nbr_clients,
        nbr_rounds,
        dataset,
    ):
        self.model = model
        self.strategy = strategy
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.dataset = dataset

    def start_server(self):
        arguments = [
            self.model,
            self.dataset["X_test"],
            self.dataset["y_test"],
            self.nbr_clients,
            self.nbr_rounds,
        ]
        eval(self.strategy + "2")(*arguments)

    def start_client(self, i, time):
        X_train_client = (
            self.dataset["X_train"][
                int((i / self.nbr_clients) * len(self.dataset["X_train"])) : int(
                    ((i + 1) / self.nbr_clients) * len(self.dataset["X_train"])
                )
            ],
        )
        y_train_client = (
            self.dataset["y_train"][
                int((i / self.nbr_clients) * len(self.dataset["y_train"])) : int(
                    ((i + 1) / self.nbr_clients) * len(self.dataset["y_train"])
                )
            ],
        )

        client = Client_Test(
            model=self.model,
            X_train=X_train_client,
            y_train=y_train_client,
            X_test=self.dataset["X_test"],
            y_test=self.dataset["y_test"],
            timed=time,
        )
        fl.client.start_numpy_client("[::]:8080", client=client)

    def run(self, time):
        process = []
        server_process = Process(
            target=self.start_server,
        )
        server_process.start()
        process.append(server_process)

        for i in range(self.nbr_clients):
            Client_i = Process(target=self.start_client, args=(i, time))
            Client_i.start()
            process.append(Client_i)

        for p in process:
            p.join()
