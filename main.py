from classes.network import Network
import pandas as pd
import numpy as np

BATCH_SIZE = 10

def shuffle_and_split(X_train, y_train, batch_size=10):
    p = np.random.permutation(len(X_train))

    new_X_train = X_train[p]
    new_y_train = y_train[p]

    splitted_X_train = np.array_split(new_X_train, batch_size)
    splitted_y_train = np.array_split(new_y_train, batch_size)

    return splitted_X_train, splitted_y_train

if __name__ == "__main__":
    network = Network(error="l2")
    network.addLayer(1, inputSize=1, activation="identity")
    #network.addLayer(1, activation="identity")

    ds = pd.read_csv("data/1in_tanh.txt", sep=r"\s+", header=None)
    X_train, y_train = ds.iloc[:,:-1].to_numpy(), ds.iloc[:,1].to_numpy()

    splitted_X_train, splitted_y_train = shuffle_and_split(X_train, y_train, batch_size=20)

    p = np.random.permutation(len(X_train))

    new_X_train = X_train[p]
    new_y_train = y_train[p]

    network.train_with_pso(new_X_train, np.expand_dims(new_y_train, axis=1), iter_count=100)

    # for (x_batch,y_batch) in zip(splitted_X_train,splitted_y_train):
    #     network.train_with_pso(x_batch, y_batch, swarm_size=100, iter_count=100)
    #     print(network.network)
    #     #print("Last mean error is: {}".format(network.train_with_pso(x, y)))
    #     print("Launching test..")
    #     print(network.test(X_train, y_train))
    print(network.predict(X_train))
    #     print("""
    #     -------------------------
    
    #     """)
