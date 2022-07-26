def partitioning(X_train, y_train,nbr_clients, epochs,accumulated_data,):
        print(epochs)
        import numpy as np
        """
        Partition the training samples 
        """
        X_train_epochs_client = [ [] for i in range(epochs)]
        y_train_epochs_client = [ [] for i in range(epochs) ]
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
                #print(len(X_train_client_epoch))
                X_train_epochs_client[epoch].append(X_train_client_epoch)
                y_train_epochs_client[epoch].append(y_train_client_epoch)

        X_train_epochs = []
        y_train_epochs = []
        print()
        print(np.array(X_train_epochs_client).shape) 
        

        for epoch in range(epochs):
            if accumulated_data:
              X_t = X_train_epochs_client[0][0]
              y_t = y_train_epochs_client[0][0]
              for k in range(epoch+1):
                for j in range(nbr_clients):
                  if (k != 0) | (j != 0): # we skip the first epoch of the first clients since it is already in the concat
                    X_t = np.concatenate([X_t, X_train_epochs_client[k][j]], 0)
                    y_t = np.concatenate([y_t, y_train_epochs_client[k][j]], 0)

            else :
              X_t = X_train_epochs_client[epoch][0]
              y_t = y_train_epochs_client[epoch][0]
              
              for i in range(1,nbr_clients):
                print(len(X_train_epochs_client[epoch][i]))
                print(i)
                X_t = np.concatenate([X_t, X_train_epochs_client[epoch][i]], 0)
                y_t = np.concatenate([y_t, y_train_epochs_client[epoch][i]], 0) 
            print(X_t.shape) 
            X_train_epochs.append(X_t)
            y_train_epochs.append(y_t)

        return X_train_epochs, y_train_epochs


def run_centralized_MNIST(epochs, nbr_clients, directory_name, accumulated_data,percentage):
    import pickle
    from Model.model_MNIST import create_model_MNIST
    from data.data_MNIST.Preprocessing_MNIST import (
        X_train,
        X_test,
        y_train,
        y_test,
    )
    import tensorflow as tf
    import time


    X_train = X_train[:int(len(X_train)*percentage)]
    y_train = y_train[:int(len(y_train)*percentage)]
    print('percentage of data used :' +str(percentage) + ' size of train data : ' + str(len(X_train) ))
    print("Accumulated data : " + str(accumulated_data))
    duration = []
    model = create_model_MNIST()
    # print(len(X_train) / 32)
    X_train, y_train = partitioning(X_train,y_train,nbr_clients , epochs ,accumulated_data)  
    
    list = []
    for epoch in range(epochs):
        print("Epoch : " + str(epoch))
        start = time.time()
        history = model.fit(
            X_train[epoch],
            y_train[epoch],
            epochs=1,
            batch_size=1,
        )
        testing_history = model.evaluate(X_test,y_test, batch_size = 64)
        end = time.time()
        duration.append(end - start)
        print("Duration : " + str(end-start))
        list.append((testing_history[1], testing_history[1]))

    for i in range(len(duration) - 1):
        duration[i + 1] += duration[i]

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)
        pickle.dump(duration, f)

