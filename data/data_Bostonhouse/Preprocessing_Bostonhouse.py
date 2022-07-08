
import tensorflow as tf 

def load_data_Bostonhouse():

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(
                path="boston_housing.npz", test_split=0.2, seed=113
                )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
  X_train, _, _ ,_ = load_data_Bostonhouse()
  print(type(X_train))
