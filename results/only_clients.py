#!/usr/bin/python3
import pickle
import os
import matplotlib.pyplot as plt
import os


def create_curves_clients(experience_path):
    dictonnary = {}
    for root, dirs, files in os.walk(experience_path + "/", topdown=True):
        print(files)
        for filename in sorted(files):
            if "client_" in filename:
                unpickleFile = open(experience_path + "/" + filename, "rb")
                new_dictonnary = pickle.load(unpickleFile, encoding="latin1")
                new_dictonnary_duration = pickle.load(unpickleFile, encoding="latin1")
                dictonnary[filename] = {}
                dictonnary[filename]["metrics"] = new_dictonnary
                dictonnary[filename]["duration"] = new_dictonnary_duration


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

    fig.savefig(experience_path + "/only_clients.png")
    fig.savefig(experience_path + "/only_clients.svg")

create_curves_clients("/home/hugo/hugo/Stage/results/25_07_2022/IMDB/non_accumulated/IMDB_FedAvg_clients_5_rounds_100_20220725173416/")

