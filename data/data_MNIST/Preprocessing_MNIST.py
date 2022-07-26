import tensorflow as tf
import numpy as np


def load_data_MNIST(nbr_clients, epochs, percentage = None):
    
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  
  if percentage is not None:
    X_train = X_train[:int(len(X_train) * percentage)]
    y_train = y_train[:int(len(y_train) * percentage)]
 
  X_train = np.expand_dims(X_train, axis = 3)
  X_test = np.expand_dims(X_test, axis =3)
  X_train = X_train/255
  X_test = X_test/255
    
  X_train, y_train = processing(
                                X_train = X_train, 
                                y_train = y_train, 
                                nbr_clients = nbr_clients, 
                                epochs = epochs)

  return X_train, X_test, y_train, y_test

def processing(X_train, y_train, nbr_clients, epochs):
        X_train_epochs_client = [ [] for i in range(nbr_clients)]
        y_train_epochs_client = [ [] for i in range(nbr_clients) ]
        for i in range(nbr_clients):
            X_train_clients = X_train[
                    int( (i / nbr_clients) * len(X_train) ) :
                    int( ( ((i +1)/nbr_clients)) * len(X_train ))
                    ]
            y_train_clients = y_train[
                    int( (i / nbr_clients) * len(y_train) ) :
                    int( ( ((i +1)/nbr_clients)) * len(y_train ))
                    ]
            for epoch in range(epochs):
                X_train_client_epoch = X_train_clients[
                    int( ( epoch / epochs) * len(X_train_clients) ):
                    int( ((epoch +1)/epochs) * len(X_train_clients) )
                ]
                y_train_client_epoch = y_train_clients[
                    int( ( epoch / epochs) * len(y_train_clients) ):
                    int( ((epoch +1)/epochs) * len(y_train_clients) )
                ]
                
                X_train_epochs_client[i].append(X_train_client_epoch)
                y_train_epochs_client[i].append(y_train_client_epoch)

        return X_train_epochs_client, y_train_epochs_client



if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data_MNIST(nbr_clients = 7,
                                                     epochs = 1,
                                                     percentage = 1)
  print(len(X_train))
  print(len(X_train[0]))
  print(len(X_train[0][0]))
