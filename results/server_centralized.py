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


    fig, ax = plt.subplots()

    for component in list(dictonnary.keys()):
        metrics = dictonnary[component]["metrics"]
        duration = dictonnary[component]["duration"]
        y = []
        for element in metrics:
            y.append(element[1])
        print(component)
        print("duration :",duration)
        print("y :",y)
        plt.plot(duration, y, label=component)
        if "JS" in experience_path:
            plt.yscale("log")

        if "Boston" in experience_path:
            plt.yscale("log")

        if "CIC_IDS" in experience_path:
            plt.legend(["server","centralized","Web_server_public","Ubuntu_server_public","Ubuntu_14_32","Ubuntu_14_64","Ubuntu_16_32","Ubuntu_16_64","Win_7","Win_8","Win_Vista","Win_10_32","Win_10_64",])

    fig.savefig(experience_path + "/server_centralized.png")
    fig.savefig(experience_path + "/server_centralized.svg")

create_curves("/home/hugo/hugo/Stage/results/25_07_2022/IMDB/non_accumulated/IMDB_FedAdam_clients_5_rounds_100_20220726011106/")

