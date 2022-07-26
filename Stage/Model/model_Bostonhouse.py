import tensorflow as tf


def create_model_Bostonhouse():
    model_Bostonhouse = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(13,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model_Bostonhouse.compile(
        metrics=[tf.keras.metrics.MeanSquaredError()],
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
    )
    return model_Bostonhouse


create_model_Bostonhouse()
