


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data_DisasterTweets():

    train_df = pd.read_csv(
        "data/data_DisasterTweets/train.csv"
    )

    # Shuffling the training dataframe
    train_df = train_df.sample(frac=1, random_state=42)
    X = np.asarray(train_df["text"])
    y = np.asarray(train_df["target"])
    
    
    X_train, X_test = train_test_split(X, test_size = 0.2, shuffle = True, random_state = 1)
    y_train, y_test = train_test_split(y, test_size = 0.2, shuffle = True, random_state = 1)

   
    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )
   
if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data_DisasterTweets()
  print(X_train.shape)
  print(X_test.shape)
  print(y_train.shape)
  print(y_test.shape)

  print(type(X_train))
  print(type(y_train))


