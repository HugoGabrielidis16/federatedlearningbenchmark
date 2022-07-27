import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

PATH = "/home/hugo/hugo/Stage/Mise_au_propre/data/data_CIC_IDS2017/"


def load_data_CIC_IDS2017():
    df_1 = pd.read_csv(
        PATH + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_unbalanced.csv"
    )
    df_2 = pd.read_csv(PATH + "Monday-WorkingHours.pcap_ISCX_unbalanced.csv")
    df_3 = pd.read_csv(PATH + "Tuesday-WorkingHours.pcap_ISCX_unbalanced.csv")
    df_4 = pd.read_csv(
        PATH + "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_unbalanced.csv"
    )
    df_5 = pd.read_csv(
        PATH + "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_unbalanced.csv"
    )
    df_6 = pd.read_csv(
        PATH + "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_unbalanced.csv"
    )
    df_7 = pd.read_csv(PATH + "Wednesday-workingHours.pcap_ISCX_unbalanced.csv")
    df_8 = pd.read_csv(PATH + "Friday-WorkingHours-Morning.pcap_ISCX_unbalanced.csv")

    df = pd.concat(
        [
            df_1,
            df_2,
            df_3,
            df_4,
            df_5,
            df_6,
            df_7,
            df_8,
        ]
    )

    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    excluded = [
        "Flow ID",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Init_Win_bytes_backward",
        "Init_Win_bytes_forward",
    ]

    y = df["Label"]
    X = df.drop(columns=["Label"])

    print(df["Label"].value_counts())

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=0.2)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    test_centralized_df = test_df.drop(
        columns=excluded + ["Timestamp"], errors="ignore"
    )

    y_test_centralized = np.array(test_centralized_df["Label"].values)
    X_test_centralized = np.array(test_centralized_df.drop(columns=["Label"]))

    Web_server_16_Public = pd.concat(
        [
            train_df[train_df["Destination IP"] == "192.168.10.50"],
            train_df[train_df["Destination IP"] == "205.174.165.68"],
        ]
    )
    Ubuntu_server_12_Public = pd.concat(
        [
            train_df[train_df["Destination IP"] == "192.168.10.51"],
            train_df[train_df["Destination IP"] == "205.174.165.66"],
        ]
    )

    Firewall = pd.concat(
        [
            train_df[train_df["Destination IP"] == "205.174.165.80"],
            train_df[train_df["Destination IP"] == "172.16.0.1"],
        ]
    )
    Ubuntu_14_4_32B = train_df[train_df["Destination IP"] == "192.168.10.19"]
    Ubuntu_14_4_64B = train_df[train_df["Destination IP"] == "192.168.10.17"]
    Ubuntu_16_4_32B = train_df[train_df["Destination IP"] == "192.168.10.16"]
    Ubuntu_16_4_64B = train_df[train_df["Destination IP"] == "192.168.10.12"]
    Win_7_Pro_64B = train_df[train_df["Destination IP"] == "192.168.10.9"]
    Win_8_1_64B = train_df[train_df["Destination IP"] == "192.168.10.5"]
    Win_Vista_64B = train_df[train_df["Destination IP"] == "192.168.10.8"]
    Win_10_pro_32B = train_df[train_df["Destination IP"] == "192.168.10.14"]
    Win_10_64B = train_df[train_df["Destination IP"] == "192.168.10.15"]
    MACe = train_df[train_df["Destination IP"] == "192.168.10.25"]

    Insiders = [
        Firewall,
        Web_server_16_Public,
        Ubuntu_server_12_Public,
        Ubuntu_14_4_32B,
        Ubuntu_14_4_64B,
        Ubuntu_16_4_32B,
        Ubuntu_16_4_64B,
        Win_7_Pro_64B,
        Win_8_1_64B,
        Win_Vista_64B,
        Win_10_pro_32B,
        Win_10_64B,
        MACe,
    ]

    for i in range(len(Insiders)):
        Insiders[i] = Insiders[i].drop(columns=excluded, errors="ignore")

        Insiders[i]["Timestamp"] = Insiders[i]["Timestamp"].apply(
            lambda x: x[1] if x[1] != "/" else x[0]  # Only keep the day
        )

    Data_per_day = []
    for i in range(len(Insiders)):
        # print(len(Insiders[i]["Timestamp"].unique())) results = 5 so 5 days
        Day_separation = []
        for j in Insiders[i]["Timestamp"].unique():
            Day_separation.append(Insiders[i][Insiders[i]["Timestamp"] == j])
        Data_per_day.append(Day_separation)

    Set = []
    for i in range(len(Data_per_day)):
        Set_i = []
        for j in range(len(Data_per_day[i])):
            Data_per_day[i][j] = Data_per_day[i][j].drop(
                columns="Timestamp", errors="ignore"
            )
            y = np.array(Data_per_day[i][j]["Label"].values)
            X_t = np.array(Data_per_day[i][j].drop(columns=["Label"]))
            Set_i.append([X_t, y])
        Set.append(Set_i)

    return Set, X_test_centralized, y_test_centralized
