import os
import shutil

import tensorflow as tf
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

def load_data_IMDB():
    pos_train_folder = (
    "data/data_IMDB/aclImdb/train/pos/"
    )
    neg_train_folder = (
    "data/data_IMDB/aclImdb/train/neg/"
    )

    X_train = []
    y_train = []

    for filename in os.listdir(pos_train_folder):
        if filename.endswith(".txt"):
            file = open(pos_train_folder + filename, "r")
            lines = file.readlines()
            X_train.append(lines)
            y_train.append(1)

    for filename in os.listdir(neg_train_folder):
        if filename.endswith(".txt"):
            file = open(neg_train_folder + filename, "r")
            lines = file.readlines()
            X_train.append(lines)
            y_train.append(0)

    for i in range(len(X_train)):
        X_train[i] = X_train[i][0]

    data = pd.DataFrame([X_train, y_train], ["text", "class"])
    data = pd.DataFrame.transpose(data)
    data = data.sample(frac=1)

    X_train = data["text"].to_numpy()
    y_train = data["class"].to_numpy()

    X_test = []
    y_test = []

    pos_test_folder = (
        "data/data_IMDB/aclImdb/test/pos/"
    )
    neg_test_folder = (
        "data/data_IMDB/aclImdb/test/neg/"
    )

    for filename in os.listdir(pos_test_folder):
        if filename.endswith(".txt"):
            file = open(pos_test_folder + filename, "r")
            lines = file.readlines()
            X_test.append(lines)
            y_test.append(1)

    for filename in os.listdir(neg_test_folder):
        if filename.endswith(".txt"):
            file = open(neg_test_folder + filename, "r")
            lines = file.readlines()
            X_test.append(lines)
            y_test.append(0)

    for i in range(len(X_test)):
        X_test[i] = X_test[i][0]

    data = pd.DataFrame([X_test, y_test], ["text", "class"])
    data = pd.DataFrame.transpose(data)
    data = data.sample(frac=1)

    X_test = data["text"].to_numpy()
    y_test = data["class"].to_numpy()
    y_train = np.asarray(y_train).astype("int")
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_data_IMDB()
  print(X_train.shape)
  print(X_test.shape)
  print(y_train.shape)
  print(y_test.shape)

  print(type(X_train))
  print(type(y_train))


