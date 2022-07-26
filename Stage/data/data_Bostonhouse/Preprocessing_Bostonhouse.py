def Data():
    import tensorflow as tf 
    import numpy as np
    import pandas as pd

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(
                path="boston_housing.npz", test_split=0.2, seed=113
                )
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = Data()

print(len(X_train))
