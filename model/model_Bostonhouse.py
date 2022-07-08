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

    return model_Bostonhouse,"mean_squared_error",tf.keras.optimizers.RMSprop(learning_rate=0.01),tf.keras.metrics.MeanSquaredError(),

