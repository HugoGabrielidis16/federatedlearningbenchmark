def run_centralized_CIC_IDS2017(epochs, nbr_clients, directory_name,accumulated_data, percentage):
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
    from sklearn.metrics import confusion_matrix
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
    
    X_test_centralized = np.reshape(X_test_centralized, (X_test_centralized.shape[0],1,X_test_centralized.shape[1]))
    
    for epoch in range(epochs):
      i = 0
      while len(Set[i]) < epoch:
       i = i + 1 
      X_epochs = Set[i][epoch][0]
      y_epochs = Set[i][epoch][1]

      for client in range(nbr_clients):
        if accumulated_data:
          for i in range(epoch+1):
            if (client != 0) | (i != epoch):
              X_epochs = np.concatenate([X_epochs, Set[client][i][0]])
              y_epochs = np.concatenate([y_epochs, Set[client][i][1]])
        else: 
          if (client !=0):
            X_epochs = np.concatenate([X_epochs, Set[client][epoch][0]])
            y_epochs = np.concatenate([y_epochs, Set[client][epoch][1]])

      #X_training.append(X_epochs)
      #y_training.append(y_epochs)

        
      #X_training[epoch] = np.reshape(X_epochs,(X_epochs.shape[0], 1, X_epochs.shape[1]))
      X_epochs = np.reshape(X_epochs,(X_epochs.shape[0], 1, X_epochs.shape[1]))
      start = time.time()
        
      train_X = X_epochs[:int(len(X_epochs) * percentage)] 
      train_y = y_epochs[:int(len(y_epochs) * percentage)]
        
      history = model.fit(
            train_X,
            train_y, 
            epochs=1,
            validation_data=(X_test_centralized, y_test_centralized),
            batch_size=32,
        )
      y_pred = model.predict(X_test_centralized)
      y_pred = np.round(y_pred)
      result = confusion_matrix(y_test_centralized, y_pred)
      print(result)
        

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
