
def partitioning(Set,epochs,accumulated_data):
    import tensorflow as tf
    import time
    import numpy as np
    from tqdm import tqdm 
    print("Start partitioning")
    X_training = []
    y_training = []
    
    for epoch in range(epochs):
      if accumulated_data:
        X_epochs = [Set[0][0][0]]
        y_epochs = [Set[0][0][1]]
        for client in tqdm(range(10)):
         for idx in range( int(len(Set[client])*((epoch+1)/epochs) ) ):
            if (client !=0) | (idx != 0):
              X_epochs.append(Set[client][idx][0])
              y_epochs.append(Set[client][idx][1])
      else:
        X_epochs = [Set[0][0][0]]
        y_epochs = [Set[0][0][1]]
        for client in tqdm(range(10)):
          for idx in range(int(len(Set[client])*(epoch/epochs)) ,int(len(Set[client])*(epoch+1)/epochs)):
            if (client !=0) | (idx != 0):
              X_epochs.append(Set[client][idx][0])
              y_epochs.append(Set[client][idx][1])
      X_training.append(X_epochs)
      y_training.append(y_epochs)
    print("finish partioning")
    return X_training, y_training



def run_centralized_MNIST_noiid(epochs, nbr_clients, directory_name,accumulated_data, percentage):
    import pickle
    from Model.model_MNIST import create_model_MNIST
    from data.data_MNIST_noiid.Preprocessing_MNIST_noiid import (Set,X_test,y_test)
    import tensorflow as tf
    import time
    import numpy as np
    model = create_model_MNIST()
    all_history = {
        "loss": [],
        "val_loss": [],
        "val_sparse_categorical_accuracy": [],
        "sparse_categorical_accuracy": [],
    }

    duration = []
    
    X_train, y_train = partitioning(Set,epochs, accumulated_data)
      
    list = []
    for epoch in range(epochs):
        
        start = time.time()
        X_epochs = X_train[epoch]
        y_epochs = y_train[epoch]

       
        train_X = np.array(X_epochs[:int(len(X_epochs) * percentage)] )
        train_y = np.array(y_epochs[:int(len(y_epochs) * percentage)])
        
        history = model.fit(
            train_X,
            train_y, 
            epochs=1,
            batch_size=1,
          )
         
        end = time.time()
        testing_history = model.evaluate(X_test,y_test)
        print(testing_history)
        duration.append(end - start)
        
    
        list.append((testing_history[1], testing_history[1]))

    for i in range(len(duration) - 1):
        duration[i + 1] += duration[i]

    file_name = directory_name + "/centralized"
    with open(file_name, "wb") as f:
        pickle.dump(list, f)
        pickle.dump(duration, f)

if __name__ == "__main__":
  from data.data_MNIST_noiid.Preprocessing_MNIST_noiid import (Set,X_test,y_test)
  X_train, y_train = partitioning(Set)

