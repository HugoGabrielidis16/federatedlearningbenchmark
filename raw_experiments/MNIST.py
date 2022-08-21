import tensorflow as tf
import numpy as np


def load_data_MNIST():

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, X_test, y_train, y_test


def create_model_MNIST():
    model_MNIST = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model_MNIST.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model_MNIST


if __name__ == "__main__":
    model = create_model_MNIST()
    model.summary()
    X_train, X_test, y_train, y_test = load_data_MNIST()
    model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=100)
