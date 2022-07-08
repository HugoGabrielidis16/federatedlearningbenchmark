
from tensorflow.keras import Sequential 

def create_model_CIC_IDS2017():
    model = Sequential()
    model.add(Input(shape=(None,74 )))
    model.add(LSTM(units=30))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid", name="sigmoid"))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='Adam',
            metrics = ["accuracy"])
    return model

