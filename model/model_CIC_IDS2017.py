import tensorflow as tf
"""
def create_model_CIC_IDS2017():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(74,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model
"""
from tensorflow.keras import Model, Sequential, Input, backend
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping



def create_model_CIC_IDS2017():
    model = Sequential()
    model.add(Input(shape=(None,74 )))
    model.add(LSTM(units=30))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid", name="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='Adam',
            metrics = ["accuracy"])
    return model

if __name__ == "__main__":
  model = create_model_CIC_IDS2017()
  model.summary()
