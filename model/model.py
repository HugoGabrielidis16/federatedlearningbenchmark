import data
from .model_IMDB import create_model_IMDB
from .model_MNIST import create_model_MNIST
from .model_DisasterTweets import create_model_DisasterTweets
from .model_Bostonhouse import create_model_Bostonhouse
from .model_CIFAR10 import create_model_CIFAR10
from .model_JS import create_model_JS


class FLModel:
    def __init__(self, dataset):
        self.model, self.loss, self.optimizer, self.metrics = eval(
            f"create_model_{dataset}"
        )
