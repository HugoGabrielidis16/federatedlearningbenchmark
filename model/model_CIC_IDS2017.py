import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense


def create_model_CIC_IDS2017():

    model = Sequential()
    model.add(Input(shape=(None, 74)))
    model.add(LSTM(units=30))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid", name="sigmoid"))

    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = "Adam"
    metrics = ["accuracy"]

    return model, loss, optimizer, metrics
