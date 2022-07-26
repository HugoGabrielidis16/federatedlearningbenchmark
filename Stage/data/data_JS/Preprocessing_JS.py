import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def Data():
    path = "/home/hugo/hugo/Stage/Data_JS/DataSet1/Data_JS/"
    df = pd.read_csv(path + "Input.csv", sep=";")
    y = pd.read_csv(path + "data_MOS.csv", sep=";").iloc[:, 0:7]

    # Preprocessing to get the Data in the (62,9,1) format
    output = [[] for i in range(7)]
    for i in y[
        ',"bspl4.1","bspl4.2","bspl4.3","bspl4.4","bspl4.5","bspl4.6","bspl4.7"'
    ]:
        l = i.split(",")
        for j in range(1, len(l)):
            output[j - 1].append(l[j])

    output = pd.DataFrame(np.transpose(output))
    y = output.astype("float64")

    df = df.drop(["Unnamed: 0"], axis=1)

    array = np.asarray(df)
    array = array.reshape([27572, 62, 9, 1])

    X_train, X_test, y_train, y_test = train_test_split(
        array, y, test_size=0.1, shuffle=False
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = Data()
print(len(X_train))
