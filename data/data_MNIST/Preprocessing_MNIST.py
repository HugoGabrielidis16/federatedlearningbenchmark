import tensorflow as tf
import numpy as np


def load_data_MNIST():
    
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  
  X_train = np.expand_dims(X_train, axis = 3)
  X_test = np.expand_dims(X_test, axis =3)
  X_train = X_train/255
  X_test = X_test/255
  return X_train, X_test, y_train, y_test




if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data_MNIST()
  print(X_train.shape)
  print(X_test.shape)
  print(y_train.shape)
  print(y_test.shape)

  print(type(X_train))
  print(type(y_train))

