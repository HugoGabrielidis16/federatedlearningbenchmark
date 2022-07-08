import tensorflow as tf


# Modele pour le DataSet de Jean-Steve


def create_model_JS():
    model_JS = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(62, 9)),
            tf.keras.layers.Dense(units=52, activation="relu"),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=512, activation="relu"),
            tf.keras.layers.Dense(units=7, activation="linear"),
        ]
    )

    model_JS.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=0.01, clipvalue=1
        ),  # clip value important to solve exploding gradient problem
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model_JS
