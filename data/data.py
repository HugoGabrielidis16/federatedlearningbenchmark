from .data_IMDB.Preprocessing_IMDB import load_data_IMDB
from .data_MNIST.Preprocessing_MNIST import load_data_MNIST
from .data_CIFAR10.Preprocessing_CIFAR10 import load_data_CIFAR10
from .data_DisasterTweets.Preprocessing_DisasterTweets import load_data_DisasterTweets
from .data_Bostonhouse.Preprocessing_Bostonhouse import load_data_Bostonhouse
from .data_JS.Preprocessing_JS import load_data_JS


class DataFactory:
    def processing(X_train, y_train, nbr_clients, epochs):
        X_train_epochs_client = [[] for i in range(nbr_clients)]
        y_train_epochs_client = [[] for i in range(nbr_clients)]
        for i in range(nbr_clients):
            X_train_clients = X_train[
                int((i / nbr_clients) * len(X_train)) : int(
                    (((i + 1) / nbr_clients)) * len(X_train)
                )
            ]
            y_train_clients = y_train[
                int((i / nbr_clients) * len(y_train)) : int(
                    (((i + 1) / nbr_clients)) * len(y_train)
                )
            ]
            for epoch in range(epochs):
                X_train_client_epoch = X_train_clients[
                    int((epoch / epochs) * len(X_train_clients)) : int(
                        ((epoch + 1) / epochs) * len(X_train_clients)
                    )
                ]
                y_train_client_epoch = y_train_clients[
                    int((epoch / epochs) * len(y_train_clients)) : int(
                        ((epoch + 1) / epochs) * len(y_train_clients)
                    )
                ]

            X_train_epochs_client[i].append(X_train_client_epoch)
            y_train_epochs_client[i].append(y_train_client_epoch)
        return X_train_epochs_client, y_train_epochs_client

    def load_data(self, dataset, nbr_clients, nbr_rounds, centralized_percentage=None):
        arguments = [nbr_clients, nbr_rounds, centralized_percentage]
        X_train, X_test, y_train, y_test = eval("load_data_" + dataset)(*arguments)
        X_train, y_train = self.processing(X_train, y_train, nbr_clients, nbr_rounds)

        return (X_train, y_train), (X_test, y_test)


# Change it to use a notion of set ? -> So we don't have to do a special case for CIC_IDS
