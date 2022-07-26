from .data_IMDB.Preprocessing_IMDB import load_data_IMDB
from .data_MNIST.Preprocessing_MNIST import load_data_MNIST
from .data_CIFAR10.Preprocessing_CIFAR10 import load_data_CIFAR10
from .data_DisasterTweets.Preprocessing_DisasterTweets import load_data_DisasterTweets
from .data_Bostonhouse.Preprocessing_Bostonhouse import load_data_Bostonhouse
from .data_JS.Preprocessing_JS import load_data_JS

class DataFactory():
    def load_data(dataset, nbr_clients, nbr_rounds, centralized_percentage = None ):
      arguments = [nbr_clients,nbr_rounds, centralized_percentage ]
      X_train, X_test, y_train, y_test = eval("load_data_" + dataset)(*arguments)

      
      return X_train, X_test, y_train, y_test  
      
