import tensorflow as tf
import numpy as np


def load_data_MNIST(nbr_clients, epochs, percentage=None):

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if percentage is not None:
        X_train = X_train[: int(len(X_train) * percentage)]
        y_train = y_train[: int(len(y_train) * percentage)]

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, X_test, y_train, y_test

    return X_train_epochs_client, y_train_epochs_client


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data_MNIST(
        nbr_clients=7, epochs=1, percentage=1
    )
    print(len(X_train))
    print(len(X_train[0]))
    print(len(X_train[0][0]))
