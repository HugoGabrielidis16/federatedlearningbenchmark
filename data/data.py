
from .data_IMDB.Preprocessing_IMDB import load_data_IMDB
from .data_MNIST.Preprocessing_MNIST import load_data_MNIST
from .data_CIFAR10.Preprocessing_CIFAR10 import load_data_CIFAR10
from .data_DisasterTweets.Preprocessing_DisasterTweets import load_data_DisasterTweets
from .data_Bostonhouse.Preprocessing_Bostonhouse import load_data_Bostonhouse


class Data():
    def __init__(self,dataset):
        self.X_train, self.X_test, self.y_train, self.y_test = eval("load_data_" + dataset)() 
