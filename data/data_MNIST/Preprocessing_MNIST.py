import tensorflow as tf
import numpy as np


def load_data_MNIST():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data_MNIST()
    print(len(X_train))
    print(len(X_train[0]))
    print(len(X_train[0][0]))
