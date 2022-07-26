import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

PATH = "/home/hugo/hugo/Stage/data/data_CIC_IDS2017/"

df_1 = pd.read_csv(PATH + "Monday-WorkingHours.pcap_ISCX_unbalanced.csv")

df_2 = pd.read_csv(PATH + "Tuesday-WorkingHours.pcap_ISCX_unbalanced.csv")

df_3 = pd.read_csv(PATH + "Wednesday-workingHours.pcap_ISCX_unbalanced.csv")
df_4 = pd.read_csv(
    PATH + "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_unbalanced.csv"
)

df_5 = pd.read_csv(
    PATH + "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX_unbalanced.csv"
)

df_6 = pd.read_csv(PATH + "Friday-WorkingHours-Morning.pcap_ISCX_unbalanced.csv")
df_7 = pd.read_csv(PATH + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX_unbalanced.csv")
df_8 = pd.read_csv(
    PATH + "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_unbalanced.csv"
)


list_df = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8]
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
#df.replace([np.inf, -np.inf], np.nan, inplace=True)
#df.dropna(inplace=True)
#print(df.info())
print(df["Label"].value_counts())
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
X = df.drop(columns=["Label"], axis =1 )

#print(df["Label"].value_counts()) 


(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.2)


print(y_test.value_counts()) 

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)


train_centralized_df = train_df
test_centralized_df = test_df

train_centralized_df = train_df.drop(columns=excluded + ["Timestamp"], errors="ignore")
test_centralized_df = test_df.drop(columns=excluded + ["Timestamp"], errors="ignore")

y_train_centralized = np.array(train_centralized_df["Label"].values)
X_train_centralized = np.array(train_centralized_df.drop(columns=["Label"]))

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
        train_df[train_df["Destination IP"] == "172.16.0.1"]
            
        ])

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
"""
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
]"""
Insiders = [
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
]
Insiders_name = [
    "Web_server_16_Public",
    "Ubuntu_server_12_Public", 
    "Ubuntu_14_4_32B",
    "Ubuntu_14_4_64B",
    "Ubuntu_16_4_32B",
    "Ubuntu_16_4_64B",
    "Win_7_Pro_64B",
    "Win_8_1_64B",
    "Win_Vista_64B",
    "Win_10_pro_32B",
    "Win_10_64B",

]
"""
Insiders_name = [
    "Firewall",
   "Web_server_16_Public",
    "Ubuntu_server_12_Public", 
    "Ubuntu_14_4_32B",
    "Ubuntu_14_4_64B",
    "Ubuntu_16_4_32B",
    "Ubuntu_16_4_64B",
    "Win_7_Pro_64B",
    "Win_8_1_64B",
    "Win_Vista_64B",
    "Win_10_pro_32B",
    "Win_10_64B",
    "MACe",

]
"""
# Keep only the data in the TimeStamp
for i in range(len(Insiders)):
    Insiders[i] = Insiders[i].drop(columns=excluded, errors="ignore")
    
    Insiders[i]["Timestamp"] = Insiders[i]["Timestamp"].apply(
        lambda x: x[1] if x[1] != "/" else x[0]  # Only keep the day
    )

# Regroup for each client data per day
Data_per_day = []
for i in range(len(Insiders)):
    print(sorted(Insiders[i]["Timestamp"].unique()))# results = 5 so 5 days
    Day_separation = []
    
    for j in sorted(Insiders[i]["Timestamp"].unique()):
        Day_separation.append(Insiders[i][Insiders[i]["Timestamp"] == j])
    Data_per_day.append(Day_separation)

# Create a set which is constitue of data per client & per day 
Set = []
for i in range(len(Data_per_day)):
    Set_i = []
    for j in range(len(Data_per_day[i])):
        Data_per_day[i][j] = Data_per_day[i][j].drop(
            columns="Timestamp", errors="ignore"
        )
        y = np.array(Data_per_day[i][j]["Label"].values).astype("float32")
        X_t = np.array(Data_per_day[i][j].drop(columns=["Label"]))
        Set_i.append([X_t, y])
    Set.append(Set_i)
# i for client, j for day, 0 for X and 1 for y

import tensorflow as tf

def count(l):
  s = 0
  for i in l:
    if i == 0:
      s+=1
  return s

if __name__ == "__main__":
    print(len(Set))
    print(len(Set[0]))
    print(len(Set[0][0]))
    days = [0,0,0,0,0]
    all_ = [0,0,0,0,0]
    days_client = [ [0,0,0,0,0] for i in range(len(Set))]
    size = [ [0,0,0,0,0] for i in range(len(Set))]
    for client in range(len(Set)):
      s  = 0
      all = 0
      for day in range(len(Set[client])):
        s += count(Set[client][day][1])
        all += len(Set[client][day][1])
        days[day] += count(Set[client][day][1])
        all_[day] += len(Set[client][day][1])
        days_client[client][day] =  count(Set[client][day][1])/len(Set[client][day][1])
        size[client][day] = len(Set[client][day][1])
      #print(s/all)
    
    for i in range(len(Set)):
      print(Insiders_name[i]+ " : "  + str(days_client[i]))
    print()
    size_per_day = [0,0,0,0,0]
    for i in range(len(size)):
      for j in range(len(size[i])):
        size_per_day[j] += size[i][j]
    
    print(size_per_day)
    print()
    for k in range(len(size)):
      print(size[k])
      #print( Insiders_name[k] +" : " +str(np.divide(np.array(size[k]),np.array(size_per_day))))
      
    

    """
    def create_model_CIC_IDS2017():
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(74,)),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(1024, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        return model
    """
    
    """
    from tensorflow.keras import Model, Sequential, Input, backend
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping


    def create_model_CIC_IDS2017():
        model = Sequential()
        model.add(Input(shape=(None,74 )))
        model.add(LSTM(units=30))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid", name="sigmoid"))
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='Adam',
            metrics = ["accuracy"])
        return model
    model = create_model_CIC_IDS2017()
    for i in range(len(Set)):
        for j in range(len(Set[i])):
            print((i, j))
            print("X size : " + str(Set[i][j][0].shape) + "y size" + str(len(Set[i][j][1])))
            print(type(X))
            Set[i][j][0] = tf.reshape(Set[i][j][0], (Set[i][j][0].shape[0],1,Set[i][j][0].shape[1]) )
            model.fit(
                Set[i][j][0],
                Set[i][j][1],
                epochs=1,
                validation_data=(X_test_centralized, y_test_centralized),
            )

    """
