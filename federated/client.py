import flwr as fl
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from copy import deepcopy

# from Launcher import timed


class Client(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        client_nbr,
        nbr_rounds,
        accumulated_data,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_nbr = client_nbr
        self.nbr_rounds = nbr_rounds
        self.accumulated_data = accumulated_data
        self.actual_rnd = 0
        self.metrics_list = []

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Fit model and return new weights as well as number of training
        examples.
        """
        self.model.set_weights(parameters)
        # Remove steps_per_epoch if you want to train over the full dataset
        # https://keras.io/api/models/model_training_apis/#fit-method
        if self.accumulated_data:
            X_train_rnd = self.X_train[0 : self.actual_rnd]
            y_train_rnd = self.y_train[0 : self.actual_rnd]

        else:
            X_train_rnd = self.X_train[self.actual_rnd]
            y_train_rnd = self.y_train[self.actual_rnd]

        self.actual_rnd += 1
        batch_size = 1
        verbose = 0
        if self.client_nbr == 0:
            verbose = 1
            self.model.fit(
                X_train_rnd,  # A modifier afin de fit pas sur les memes donnes (le client genere des donnes sucessivent)
                y_train_rnd,
                epochs=config["local_epochs"],
                batch_size=batch_size,
                verbose=verbose,
            )
        testing_history = self.model.evaluate(self.X_test, self.y_test, verbose=verbose)
        loss, metrics_used = testing_history

        print(
            " ------------------------------------------------ Client number "
            + str(self.client_nbr)
            + " results - loss :"
            + str(loss)
            + " -  other metrics : "
            + str(metrics_used)
            + " ------------------------------------------------"
        )
        self.metrics_list.append(testing_history)

        # self.evaluate(parameters)
        return self.model.get_weights(), len(X_train_rnd), {}

    # This function seems to not be call

    def evaluate(self, parameters):
        "Evaluate using provided parameters."
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the test data
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=1)

        return loss, len(self.X_test), {"metrics": accuracy}
