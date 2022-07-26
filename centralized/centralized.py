import pickle
import time 
import tensorflow as tf
import numpy as np
from multiprocessing import Process
#tf.disable_v2_behavior()
from data.data import DataFactory



class Centralized(Process):

    def __init__(self, model, dataset, nbr_clients, nbr_rounds, directory_name, metrics, accumulated_data, percentage):
        super(Centralized,self).__init__()
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = DataFactory.load_data(dataset,nbr_clients, nbr_rounds, percentage )
        
        self.nbr_clients = nbr_clients
        self.epochs = nbr_rounds
        self.accumulated_data = accumulated_data
        self.directory_name = directory_name + "/centralized"
        self.metrics_list = []
               
             
        self.duration = []
        

    def partitioning(self):
        """
        Partition the training samples 
        """
        X_train_epochs = []
        y_train_epochs = []
        for epoch in range(self.epochs):
            X_t = self.X_train[epoch][0]
            y_t = self.y_train[epoch][0]

            if self.accumulated_data:
              for k in range(epoch+1):
                for j in range(self.nbr_clients):
                  if (k == 0 ) & ( j == 0): # we skip the first epoch of the first clients since it is already in the concat
                    pass
                  else : 
                    X_t = np.concatenate([X_t, self.X_train[k][j]], 0)
                    y_t = np.concatenate([y_t, self.y_train[k][j]], 0)

            else :

              for i in range(1,len(self.X_train[epoch])):
                X_t = np.concatenate([X_t, self.X_train[epoch][i]], 0)
                y_t = np.concatenate([y_t, self.y_train[epoch][i]], 0) 
            
            X_train_epochs.append(X_t)
            y_train_epochs.append(y_t)

        return X_train_epochs, y_train_epochs

    def saving(self):
        """
        After the training, modify the duration and save in pickle format the duration and the metrics
        """
        for i in range(len(self.duration)-1):
            self.duration[i+1] += self.duration[i]
        
        metrics_list = []
        # To change and incorporate a list of val_metrics and val_loss


        with open(self.directory_name, "wb") as f:
            pickle.dump(self.metrics_list, f)
            pickle.dump(self.duration, f)
        

    def run(self):   
        """
        Partition X_train & y_train in the way we dit for flower clients,
        run the training and save the results in a pickle
        """
        X_train_epochs, y_train_epochs = self.partitioning() 
        for epoch in range(self.epochs):
          start = time.time()

          history = self.model.fit( 
                (X_train_epochs[epoch]), 
                (y_train_epochs[epoch]), 
                batch_size = 1,
                )
          loss, metrics_used = self.model.evaluate(self.X_test, self.y_test, batch_size = 32, verbose = 1)
          self.metrics_list.append((loss,metrics_used))
          end = time.time()
          self.duration.append(end-start)
        
        self.saving()
