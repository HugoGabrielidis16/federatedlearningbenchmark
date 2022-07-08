def run_centralized_CIC_IDS2017(epochs, nbr_clients, directory_name):
    import pickle
    from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017
    from data.data_CIC_IDS2017.experiment import (
        Set,
        X_test_centralized,
        y_test_centralized,
    )
    import tensorflow as tf
    import time
    import numpy as np

    model = create_model_CIC_IDS2017()
    all_history = {
        "loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "accuracy": [],
    }

    X_training = []
    y_training = []
    duration = []

    X_test_centralized = tf.reshape(X_test_centralized,(X_test_centralized.shape[0], 1, X_test_centralized.shape[1]))
    for i in range(nbr_clients):
        X_epochs = Set[i][0][0]
        y_epochs = Set[i][0][1]
        for j in range(1, epochs):
            X_epochs = tf.concat([X_epochs, Set[i][j][0]], 0)
            y_epochs = tf.concat([y_epochs, Set[i][j][1]], 0)

        X_training.append(X_epochs)
        y_training.append(y_epochs)

    for i in range(epochs):
        
        
        X_training[i] = tf.reshape(X_training[i],(X_training[i].shape[0], 1, X_training[i].shape[1]))
        start = time.time()
        history = model.fit(
            X_training[i],
            y_training[i],
            epochs=1,
            validation_data=(X_test_centralized, y_test_centralized),
            batch_size=32,
        )

        end = time.time()
        duration.append(end - start)
        for key in history.history.keys():
            all_history[key].append(history.history[key])
    list = []
    for key in all_history.keys():
        if "val" in key and "loss" not in key:  # ugly way to only select the metrics
            for i in range(len(all_history[key])):
                list.append((all_history[key][i], all_history[key][i]))

    for i in range(len(duration) - 1):
        duration[i + 1] += duration[i]

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)
        pickle.dump(duration, f)
