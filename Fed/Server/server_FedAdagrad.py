from multiprocessing import Process

import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pickle
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes
import time


def get_eval_fn(model2, X_test, y_test, list_metrics, duration):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model2 here to avoid the overhead of doing it in `evaluate` itself

    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        
        duration.append(time.time())
        model2.set_weights(weights)  # Update model2 with the latest parameters
        
        loss, metrics_used = model2.evaluate(X_test, y_test)
        list_metrics.append((loss, metrics_used))
        print("Test after evaluate")
        return loss, {"other metrics": metrics_used}  # ,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {"batch_size": 32, "local_epochs": 1 if rnd < 2 else 2, "rnd": rnd}
    return config


def evaluate_config(rnd: int):
    print("evaluate_config")
    """Return evaluation configuration dict for each round.

            Perform five local evaluation steps on each client (i.e., use five
            batches) during rounds one to three, then increase to ten local
            evaluation steps.
            """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}


class FedAdagrad2(Process):
    def __init__(self, model2, X_test, y_test, nbr_clients, nbr_rounds, directory_name):
        print("Test init")
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.model = model2
        self.directory_name = directory_name
        # self.model = create_model_JS()
        self.duration = [time.time()]
        self.run()

    def run(self):

        list_metrics = []
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=self.nbr_clients,
            min_eval_clients=self.nbr_clients,
            min_available_clients=self.nbr_clients,
            eval_fn=get_eval_fn(
                self.model, self.X_test, self.y_test, list_metrics, self.duration
            ),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=None,
            eta=0.1,
            tau=0.01,
        )
        print("Before server")
        fl.server.start_server(
            "[::]:8080", config={"num_rounds": self.nbr_rounds}, strategy=strategy
        )

        file_name = self.directory_name + "/server"
        list = []
        for i in range(len(self.duration) - 1):
            list.append(self.duration[i + 1] - self.duration[i])
        list.pop(0)
        for i in range(len(list) - 1):
            list[i + 1] += list[i]

        with open(file_name, "wb") as f:
            pickle.dump(list_metrics, f)
            pickle.dump(list, f)
