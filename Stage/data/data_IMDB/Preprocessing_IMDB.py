import os
import shutil

import tensorflow as tf
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
"""
tf.get_logger().setLevel("ERROR")

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file(
    "aclImdb_v1.tar.gz", url, untar=True, cache_dir=".", cache_subdir=""
)

dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")

train_dir = os.path.join(dataset_dir, "train")

# remove unused folders to make it easier to load the data
# remove_dir = os.path.join(train_dir, "unsup")
# shutil.rmtree(remove_dir)
"""
pos_train_folder = (
    "/home/hugo/hugo/Stage//data/data_IMDB/aclImdb/train/pos/"
)
neg_train_folder = (
    "/home/hugo/hugo/Stage/data/data_IMDB/aclImdb/train/neg/"
)


def Data():
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
        "/home/hugo/hugo/Stage/data/data_IMDB/aclImdb/test/pos/"
    )
    neg_test_folder = (
        "/home/hugo/hugo/Stage/data/data_IMDB/aclImdb/test/neg/"
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

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = Data()


X_train = tf.convert_to_tensor(X_train)

y_train = np.asarray(y_train).astype("int")
y_train = tf.convert_to_tensor(y_train)

X_test = tf.convert_to_tensor(X_test)

y_test = np.asarray(y_test).astype("int")
y_test = tf.convert_to_tensor(y_test)
