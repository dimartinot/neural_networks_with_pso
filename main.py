from classes.network import Network
import pandas as pd

if __name__ == "__main__":
    network = Network(error="l1")
    network.addLayer(1, inputSize=1, activation="identity")

    ds = pd.read_csv("data/1in_linear.txt", sep=r"\s+", header=None)
    x, y = ds.iloc[:,:-1].to_numpy(), ds.iloc[:,1].to_numpy()

    print("Last mean error is: {}".format(network.train_with_pso(x, y)))
    print("Launching test..")
    print(network.test(x, y))
    print(network.predict(x))
