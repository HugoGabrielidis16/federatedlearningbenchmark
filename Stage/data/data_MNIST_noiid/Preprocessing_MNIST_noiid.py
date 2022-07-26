def Data():
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import random
    from copy import deepcopy

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = tf.reshape(X_train, X_train.shape + (1,))
    X_test = tf.reshape(X_test, X_test.shape + (1,))
    X_train = X_train / 255
    X_test = X_test / 255
    

    labels = [ [] for i in range(10)]

    for idx, value in enumerate(y_train):
      labels[value].append(idx)
    
    # labels : 0 -> 59 999, 0 : 6000 -> 0
    Set = [ [] for client in range(10) ]
    

    # --> de 0 a 3000
    # Client_0 : 0 1 
    # Client 1 : 2 3
    # Client 2 : 4 5
    # Client 3 : 6 7 
    # Client 4 : 8 9
    # --> de 3001 a 6000
    # Client 5 : 0 9
    # Client 6 : 1 8
    # Client 7 : 2 7
    # Client 8 : 3 6
    # Client 9 : 4 5
    client_list = [ [0,1], [2,3], [4,5], [6,7], [8,9], [0,9], [1,8], [2,7], [3,5], [4,6] ]
    
    for client in range(len(Set)):
      decalage = False
      if client >= 5:
        decalage = True

      value_list = client_list[client]

      for value in value_list:
        if not decalage:
          for idx in range(0, round(len(labels[value])/2)):
            Set[client].append( [X_train[labels[value][idx]], value])
        else :
          for idx in range( round(len(labels[value])/2) +1, len(labels[value]) ):
              Set[client].append( [X_train[labels[value][idx]], value])
    

    # Shuffle the differents clients dataset 
    for client in range(len(Set)):
      a = deepcopy(Set[client])
      np.random.shuffle(a)
      Set[client] = a
    print("finish loading the data..." )   
    return Set, X_test, y_test 

Set, X_test, y_test = Data()

if __name__ == "__main__":
  print(Set[0][2][1].shape)

