import flwr as fl
import tensorflow as tf

# Load and compile Keras model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# Define Flower client
class MNISTClient_test(fl.client.NumPyClient):
    def __init__(self, X_train, X_test, y_train, y_test,model):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    
        

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        self.model.set_weights(parameters)
        # Remove steps_per_epoch if you want to train over the full dataset
        # https://keras.io/api/models/model_training_apis/#fit-method

        history = self.model.fit(
            self.X_train,  # A modifier afin de fit pas sur les memes donnes (le client genere des donnes sucessivent)
            self.y_train,
            epochs=1,
            batch_size=3,
            steps_per_epoch=1,
            verbose=1,
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):

        """Evaluate using provided parameters."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}
    
    def get_properties():
        pass


