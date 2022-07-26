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
    
    for client in range(nbr_clients):

        X_t = X_train_epochs[0][0]

        for actual_rnd in range(epochs):
            for k in range(actual_rnd):
                if client == 0 & k == 0 :
                    pass
                else:
                    X_t = tf.concat([X_train_epochs[client][k], X_t], axis = 0)

run_centralized_JS(5,5,"test")
