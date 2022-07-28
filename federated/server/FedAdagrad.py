from multiprocessing import Process
import time
import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
import pickle
from flwr.server.client_proxy import ClientProxy
from copy import deepcopy


def get_eval_fn(model, X_test, y_test, list_metrics, duration):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    # The `evaluate` function will be called after every round

    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        duration.append(time.time())
        model.set_weights(weights)  # Update model2 with the latest parameters

        loss, metrics_used = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
        list_metrics.append((loss, metrics_used))
        print(
            " ------------------------------------------------ Server results - loss :"
            + str(loss)
            + " -  other metrics : "
            + str(metrics_used)
            + " duration : "
            + str(duration[len(duration) - 1] + duration[len(duration) - 1])
            + " ------------------------------------------------"
        )
        return loss, {"other metrics": metrics_used}  # ,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {"batch_size": 32, "local_epochs": 1, "rnd": rnd}
    return config


def evaluate_config(rnd: int):
    print("evaluate_config")
    """Return evaluation configuration dict for each round.

            Perform five local evaluation steps on each client (i.e., use five
            batches) during rounds one to three, then increase to ten local
            evaluation steps.
            """
    val_steps = 1
    return {"val_steps": val_steps}


class FedAdagrad(Process):
    def __init__(self, model, X_test, y_test, nbr_clients, nbr_rounds, directory_name):

        super(FedAdagrad, self).__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.model = model
        self.filename = directory_name + "/server"

        self.duration = [time.time()]
        self.metrics_list = []
        self.run()

    def saving(self):
        time_list = []

        for i in range(len(self.duration) - 1):
            time_list.append(self.duration[i + 1] - self.duration[i])
        time_list.pop(0)

        for i in range(len(time_list) - 1):
            time_list[i + 1] += time_list[i]

        with open(self.filename, "wb") as f:
            pickle.dump(self.metrics_list, f)
            pickle.dump(time_list, f)

    def run(self):

        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit=1,
            fraction_eval=1,
            min_fit_clients=self.nbr_clients,
            min_eval_clients=self.nbr_clients,
            min_available_clients=self.nbr_clients,
            eval_fn=get_eval_fn(
                self.model,
                self.X_test,
                self.y_test,
                self.metrics_list,
                self.duration,
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

        self.saving()
