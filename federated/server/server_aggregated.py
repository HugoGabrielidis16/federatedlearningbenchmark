import flwr as fl
from typing import Any, Callable, Dict, List, Optional, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.common import EvaluateRes


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy


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
        print("Test after evaluate")
        return loss, {"other metrics": metrics_used}  # ,loss ( not really needed )

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {"batch_size": 32, "local_epochs": 1}
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


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(
            f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}"
        )

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


def run_aggregated_server(nbr_clients, model, X_test, y_test, nbr_rounds):
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1,
        fraction_eval=0.2,
        min_fit_clients=nbr_clients,
        min_eval_clients=2,
        min_available_clients=nbr_clients,
        eval_fn=get_eval_fn(model, X_test, y_test),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    fl.server.start_server(
        "[::]:8080", config={"num_rounds": nbr_rounds}, strategy=strategy
    )
