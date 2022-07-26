#!/usr/bin/python3
import pickle
import os
import matplotlib.pyplot as plt
import os


def create_curves(experience_path):

    dictonnary = {"server": {}, "centralized": {}}
    for root, dirs, files in os.walk(experience_path + "/", topdown=True):
        print(files)
        for filename in sorted(files):

            if filename == "server":
                unpickleFile = open(experience_path + "/" + filename, "rb")
                server_metrics = pickle.load(unpickleFile, encoding="latin1")
                server_duration = pickle.load(unpickleFile, encoding="latin1")
                server_metrics.pop(
                    0
                )  # We skip the evaluation round that happend before training
                dictonnary["server"]["metrics"] = server_metrics
                dictonnary["server"]["duration"] = server_duration

            elif filename == "centralized":
                unpickleFile = open(experience_path + "/" + filename, "rb")
                centralized_metrics = pickle.load(unpickleFile, encoding="latin1")
                centralized_duration = pickle.load(unpickleFile, encoding="latin1")

                dictonnary["centralized"]["metrics"] = centralized_metrics
                dictonnary["centralized"]["duration"] = centralized_duration

            elif "client_" in filename:
                unpickleFile = open(experience_path + "/" + filename, "rb")
                new_dictonnary = pickle.load(unpickleFile, encoding="latin1")
                dictonnary[filename] = {}
                dictonnary[filename]["metrics"] = new_dictonnary

    fig, ax = plt.subplots()

    for component in list(dictonnary.keys()):
        if "client_" in component:
            dictonnary[component]["duration"] = dictonnary["server"]["duration"]
        #print(component)
        #print(dictonnary[component])
        metrics = dictonnary[component]["metrics"]
        duration = dictonnary[component]["duration"]
        #print(duration)
        #print((component, len(metrics), len(duration)))
        y = []
        for element in metrics:
            y.append(element[1])
        print(component)
        print("duration :",duration)
        print("y :",y)
        plt.plot(duration, y, label=component)

        if "JS" in experience_path:
            #ax.set_ylim([0,5])
            plt.yscale("log")
        if "Boston" in experience_path:
            #ax.set_ylim([0,100])
            plt.yscale("log")
        if "CIC_IDS" in experience_path:
            #plt.legend(["server","centralized","Firewall","Web_server_public","Ubuntu_server_public","Ubuntu_14_32","Ubuntu_14_64","Ubuntu_16_32","Ubuntu_16_64","Win_7","Win_8","Win_Vista","Win_10_32","Win_10_64","MACe"])
            plt.legend(["server","centralized","Web_server_public","Ubuntu_server_public","Ubuntu_14_32","Ubuntu_14_64","Ubuntu_16_32","Ubuntu_16_64","Win_7","Win_8","Win_Vista","Win_10_32","Win_10_64",])
        # plt.legend()

    fig.savefig(experience_path + "/metric_with_all_client.png")
    fig.savefig(experience_path + "/metric_with_all_client.svg")

create_curves("IMDB_FedAvg_clients_5_rounds_100_20220722140017/")

