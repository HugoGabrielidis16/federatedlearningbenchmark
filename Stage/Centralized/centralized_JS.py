def run_centralized_JS(epochs, nbr_clients, directory_name):
    import tensorflow as tf
    import pickle
    from Model.model_JS import create_model_JS
    from data.data_JS.Preprocessing_JS import X_test, X_train, y_test, y_train
    import time

    model = create_model_JS()
    X_train_epochs = [[] for w in range(epochs)]
    y_train_epochs = [[] for w in range(epochs)]
    all_history = {
        "loss": [],
        "val_loss": [],
        "val_mean_squared_error": [],
        "mean_squared_error": [],
    }
    for i in range(nbr_clients):
        X_train_i = X_train[
            int((i / nbr_clients) * len(X_train)) : int(
                ((i + 1) / nbr_clients) * len(X_train)
            )
        ]
        y_train_i = y_train[
            int((i / nbr_clients) * len(y_train)) : int(
                ((i + 1) / nbr_clients) * len(y_train)
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
    # print(len(X_train_epochs[0][0]))
    # print(len(X_train_epochs[1][0]))

    # print("SIZEE = " + str(len(X_train_epochs[0])))
    duration = []
    for i in range(epochs):
        X_t = X_train_epochs[0][0]
        y_t = y_train_epochs[0][0]
        # print("Size of X_train_epochs :" + str(len(X_train_epochs[i])))
        # print(type(X_t))
        for k in range(i+1):
            for j in range(nbr_clients):
                if ( k == 0 ) & ( j == 0):
                    pass
                else :
                    X_t = tf.concat([X_t, X_train_epochs[k][j]], 0)
                    y_t = tf.concat([y_t, y_train_epochs[k][j]], 0)
        start = time.time()
        history = model.fit(
            X_t, y_t, epochs=1, validation_data=(X_test, y_test), batch_size=32
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
