def Data():
    import tensorflow as tf
    import numpy as np
    import pandas as pd

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = tf.reshape(X_train, X_train.shape + (1,))
    X_test = tf.reshape(X_test, X_test.shape + (1,))
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = Data()
