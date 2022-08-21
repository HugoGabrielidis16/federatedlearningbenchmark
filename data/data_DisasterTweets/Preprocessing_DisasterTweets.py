def Data():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    train_df = pd.read_csv("data/data_DisasterTweets/train.csv")
    # Shuffling the training dataframe
    train_df_shuffled = train_df.sample(frac=1, random_state=42)

    # Split our data into training and test sets
    data_X = np.asarray(train_df_shuffled["text"])
    y = np.asarray(train_df_shuffled["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        data_X,
        y,
        test_size=0.1,
    )
    return (
        tf.constant(X_train[: int(len(X_train))]),
        tf.constant(X_test),
        tf.constant(y_train[: int(len(y_train))]),
        tf.constant(y_test),
    )


X_train, X_test, y_train, y_test = Data()
print(len(X_train))
