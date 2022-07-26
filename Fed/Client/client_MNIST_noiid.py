import os
import flwr as fl
import tensorflow as tf
import numpy as np


class Client_MNIST_noiid(fl.client.NumPyClient):
    def __init__(self, model, Set, X_test, y_test, client_nbr, total_rnd, accumulated_data):
        self.model = model
        self.Set = Set
        self.X_test = X_test
        self.y_test = y_test
        self.client_nbr = client_nbr
        self.metrics_list = []
        self.total_rnd = total_rnd
        self.actual_rnd = 0
        self.accumulated_data = accumulated_data

    def get_parameters(self):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        self.model.set_weights(parameters)

        epochs: int = config["local_epochs"]
        batch_size =1 
        
        if self.accumulated_data:
          X = [self.Set[0][0]]
          y = [self.Set[0][1]]
          for idx in range(1, int(len(self.Set) * (self.actual_rnd+1)/self.total_rnd )):
            X.append(self.Set[idx][0])
            y.append(self.Set[idx][1])
        else:
          X = [self.Set[int(len(self.Set) * self.actual_rnd/self.total_rnd)][0]]
          y = [self.Set[int(len(self.Set) * self.actual_rnd/self.total_rnd)][1]]
          for idx in range(int(len(self.Set) * self.actual_rnd/self.total_rnd)+1, int(len(self.Set) * (self.actual_rnd+1)/self.total_rnd )):
            X.append(self.Set[idx][0])
            y.append(self.Set[idx][1])
        
        X = np.array(X)
        y = np.array(y)
        self.model.fit(
            x=X,
            y=y,  # In each round the client will train with differents data
            batch_size=batch_size,
            epochs=1,
            verbose=0,
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
