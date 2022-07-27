from .data_IMDB.Preprocessing_IMDB import load_data_IMDB
from .data_MNIST.Preprocessing_MNIST import load_data_MNIST
from .data_CIFAR10.Preprocessing_CIFAR10 import load_data_CIFAR10
from .data_DisasterTweets.Preprocessing_DisasterTweets import load_data_DisasterTweets
from .data_Bostonhouse.Preprocessing_Bostonhouse import load_data_Bostonhouse
from .data_JS.Preprocessing_JS import load_data_JS


class DataFactory:
    def processing(self, X_train, y_train, nbr_clients, epochs):
        """
        Change X_train, y_train format to Set_X & Set_y.
        Where Set is organised following the format:
        Set[i][j] : Set of training of the client number "i" for the
        rounds "j".

        Can be passed as it is to the federated.py.

        Args
        --------
        X_train (tensor, array) :
        y_train (tensor, array) :
        nbr_clients (int) : the number of cliens
        epochs (int) : the number of training rounds

        Returns
        --------
        Set_X (matrix) :
        Set_y (matrix) :
        """

        Set_X = [[0 for epoch in range(epochs)] for client in range(nbr_clients)]
        Set_y = [[0 for epoch in range(epochs)] for client in range(nbr_clients)]

        for client in range(nbr_clients):
            X_train_clients = X_train[
                int((client / nbr_clients) * len(X_train)) : int(
                    (((client + 1) / nbr_clients)) * len(X_train)
                )
            ]
            y_train_clients = y_train[
                int((client / nbr_clients) * len(y_train)) : int(
                    (((client + 1) / nbr_clients)) * len(y_train)
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

            Set_X[client] = X_train_client_epoch
            Set_y[client] = y_train_client_epoch
        return Set_X, Set_y

    def load_data(self, dataset, nbr_clients, nbr_rounds):
        """
        Load the data of the target dataset and partition it according
        to the number of clients and rounds
        """

        arguments = [nbr_clients, nbr_rounds]
        X_train, X_test, y_train, y_test = eval("load_data_" + dataset)(*arguments)
        X_train, y_train = self.processing(X_train, y_train, nbr_clients, nbr_rounds)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }


if __name__ == "__main__":
    dataset = DataFactory.load_data("MNIST")
