import flwr as fl

from Federated.server.FedAvg import FedAvg
from Federated


class Federated():

    def __init__(self, ,model, X_train , X_test, y_train ,y_test,strategy  ,nbr_clients, nbr_rounds, directory_name):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.strategy = strategy
        self.nbr_clients = nbr_clients
        self.nbr_rounds = nbr_rounds
        self.directory_name = directory_name
    
    def start_server(self):
        """
        Start a process for the server, call the class associated to the strategy
        """
        arguments = [self.model, self.X_test, self.nbr_clients, self.nbr_rounds, self.directory_name]
        server = eval(strategy ,arguments)
    
    def start_client(self, X_train_client, y_train_client, client_nbr):
        """
        Start a process for a single client with it associated dataset, dump the results in a pickle
        """
        client = Client_Test(
                model = self.model,
                X_train = X_train_client,
                y_train = y_train_client,
                X_test = self.X_test,
                y_test = self.y_test,
                client_nbr = client_nbr,
                nbr_rounds = self.nbr_rounds,
                )
        fl.client.start_numpy_client("[::]:8080", client = client)
        filename = self.directory_name + "/client_number_" + str(client_nbr)
    
        with open(filename, "wb") as f:
            pickle.dump(client.metrics_list, f)

        def run(self):
            """
            Run the experience, with the server and each clients as a subprocess. The results will be dump in
            a pickle for each one
            """
            process = []
            server_process = Process(
                    target = self.start_server,
                    args = (self),
                    )
            server_process.start()
            process.append(server_process)
            time.sleep(5)

            for i in range(self.nbr_clients):
                X_train_client = self.X_train[
                        int(( i / self.nbr_clients) * len(self.X_train)) : 
                        int( ((i+1) / self.nbr_clients) *len(self.X_train)) ]
                
                y_train_client = self.y_train[
                        int(( i / self.nbr_clients) * len(self.y_train)) : 
                        int( ((i+1) / self.nbr_clients) *len(self.y_train)) ]
                Client_i = Process(
                        target = self.start_client,
                        args = (self, X_train_client, y_train_client, i),
                        )
                Client_i.start()
                process.append(Client_i)
            
            for subprocess in process :
                subprocess.join()

