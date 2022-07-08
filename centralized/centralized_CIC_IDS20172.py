def run_centralized_CIC_IDS2017(epochs, nbr_clients, directory_name):
    import pickle
    from Model.model_CIC_IDS2017 import create_model_CIC_IDS2017
    from data.data_CIC_IDS2017.experiment import (
        X_train_centralized,
        X_test_centralized,
        y_train_centralized,
        y_test_centralized,
    )
    import tensorflow as tf
    import time

    model = create_model_CIC_IDS2017()
    X_train_epochs = [[] for w in range(epochs)]
    y_train_epochs = [[] for w in range(epochs)]
    all_history = {
        "loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "accuracy": [],
    }
    print(f"""size of X_train_centralized {len(X_train_centralized)} """)
    for i in range(nbr_clients):

        X_train_i = X_train_centralized[
            int((i / nbr_clients) * len(X_train_centralized)) : int(
                ((i + 1) / nbr_clients) * len(X_train_centralized)
            )
        ]
        y_train_i = y_train_centralized[
            int((i / nbr_clients) * len(y_train_centralized)) : int(
                ((i + 1) / nbr_clients) * len(y_train_centralized)
            )
        ]
        # So each client have a different dataset to train on
        for actual_rnd in range(epochs):
            X_train_i_actual_rnd = X_train_i[
                int((actual_rnd / epochs) * len(X_train_i)) : int(
                    ((actual_rnd + 1) / epochs) * len(X_train_i)
                )
            ]
            y_train_i_actual_rnd = y_train_i[
                int((actual_rnd / epochs) * len(y_train_i)) : int(
                    ((actual_rnd + 1) / epochs) * len(y_train_i)
                )
            ]
            X_train_epochs[actual_rnd].append(X_train_i_actual_rnd)
            y_train_epochs[actual_rnd].append(y_train_i_actual_rnd)

    print(f"""size of X_train_epochs {len(X_train_epochs)} """)
    duration = []

    for i in range(epochs):
        X_t = X_train_epochs[i][0]
        y_t = y_train_epochs[i][0]

        for j in range(1, len(X_train_epochs[i])):

            X_t = tf.concat([X_t, X_train_epochs[i][j]], 0)
            y_t = tf.concat([y_t, y_train_epochs[i][j]], 0)

        start = time.time()

        print(f"""size of X_train_epochs {len(X_t)} """)
        history = model.fit(
            X_t,
            y_t,
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
