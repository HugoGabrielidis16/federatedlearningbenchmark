from multiprocessing import Process
import flwr as fl
from typing import Dict, Optional, Tuple
from flwr.server.strategy import *
import matplotlib.pyplot as plt

list_metrics = []


def get_eval_fn(model, X_test, y_test):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # Use the last 5k training examples as a validation set

    # The `evaluate` function will be called after every round

    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        model.set_weights(weights)  # Update model with the latest parameters
        print("Test evaluate")
        # model.summary()
        # model.fit(X_test, y_test, epochs=5)
        loss, metrics_used = model.evaluate(X_test, y_test)
        list_metrics.append(metrics_used)
        print("Test after evaluate")
        return loss, {"other metrics": metrics_used}  # ,loss ( not really needed )

    return evaluate


def evaluate_config(rnd: int):
    print("evaluate_config")
    """Return evaluation configuration dict for each round.

            Perform five local evaluation steps on each client (i.e., use five
            batches) during rounds one to three, then increase to ten local
            evaluation steps.
            """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}


class Server(Process):
    def init(self, strategy, model, X_test, y_test, nbr_clients, nbr_rounds):
        super().__init__()
        self.strategy = strategy
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.nbr_client = nbr_clients
        self.nbr_rounds = nbr_rounds

    def run(self):

        fl.server.start_server(
            "[::]:8080", config={"num_rounds": self.nbr_rounds}, strategy=self.strategy
        )
