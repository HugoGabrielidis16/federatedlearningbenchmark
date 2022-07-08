import os
import flwr as fl
import tensorflow as tf
import numpy as np


class Client_CIC_IDS2017(fl.client.NumPyClient):
    def __init__(self, model, Set, X_test, y_test, client_nbr, timed, total_rnd):
        self.model = model
        self.Set = Set
        self.X_test = X_test
        self.y_test = y_test
        self.client_nbr = client_nbr
        self.timed = timed
        self.metrics_list = []
        self.total_rnd = total_rnd
        self.actual_rnd = 0

    def get_parameters(self):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        self.model.set_weights(parameters)

        epochs: int = config["local_epochs"]
        batch_size = 32
        X = np.array(self.Set[self.actual_rnd][0], dtype=np.uint8)
        y = np.array(self.Set[self.actual_rnd][1], dtype=np.uint8)
        y = y.reshape((-1, 1))
        print(type(X))

        X = tf.reshape(X, (X[0].shape,1,X[1].shape))
        self.model.fit(
            x=X,
            y=y,  # In each round the client will train with differents data
            batch_size=batch_size,
            epochs=1,
            verbose=1,
        )
        self.actual_rnd += 1
        testing_history = self.model.evaluate(self.X_test, self.y_test)
        self.metrics_list.append(testing_history)

        try:
            cardinal = len(self.Set[self.actual_rnd])
        except:
            cardinal = 0

        return self.model.get_weights(), cardinal, {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        self.model.set_weights(parameters)

        steps: int = config["val_steps"]

        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}
