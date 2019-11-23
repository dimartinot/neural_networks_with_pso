import numpy as np
import pandas as pd

from random import seed
from random import random

from classes.utils.utils import *
from classes.pso import PSO

from tqdm import tqdm

class Network:

    alpha = 0.8

    network = {
        "weights": {},
        "biases": {},
        "activationFunctions":{},
        "fp_activation": {},
        "deltas": {}}

    activationFunctions={
            #"identity": lambda x: x,
            #"binary_step": lambda x: 0 if x<0 else 1,
            "logistic": lambda x: 1/(1+np.exp(-x)),
            "tanh": lambda x: np.tanh(x),
            #"arctan": lambda x: np.arctan(x),
            #"relu": lambda x: 0 if x<0 else x,
            "cosine": lambda x: np.cos(x),
            "gaussian": lambda x: np.exp(-(x**2)/2),
            #"prelu": lambda x: Network.alpha*x if x<0 else x,
            #"elu": lambda x: Network.alpha*(np.exp(x)-1) if x<0 else x,
            #"softplus" : lambda x: np.log(1+np.exp(x)),
            
            #"d_identity": lambda x: 1,
            #"d_binary_step": lambda x: 0 if x!=0 else unknownDerivative(),
            "d_logistic": lambda x: (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x)))),
            "d_tanh": lambda x: 1-(np.tanh(x))**2,
            #"d_arctan": lambda x: 1/((x**2)-1),
            #"d_relu": lambda x: 0 if x<0 else 1,
            "d_cosine": lambda x: np.sin(x),
            "d_gaussian": lambda x: -x*np.exp(-(x**2)/2),
            #"d_prelu": lambda x: Network.alpha if x<0 else 1,
            #"d_elu": lambda x: Network.activationFunctions["prelu"](x)+Network.alpha if x<0 else 1,
            #"d_softplus" : lambda x: 1/(1+np.exp(-x)),
            
            }

    np.random.seed(42)
    seed(42)

    def __init__(self, error="l1"):
        self.network = {
            "weights": {},
            "biases":{},
            "activationFunctions":{},
            "fp_activation": {},
            "deltas": {}}

        if error == "l1":
            self.calcError = l1_error
        else:
            self.calcError = l2_error

    def getNumberOfLayers(self):
        return len(self.network["weights"].keys())

    def getActivation(self, activation_key):
        try:
            return activation_key.lower()
        except:
            return "tanh"

    def addLayerToNetwork(self, layer, activation="tanh"):
        
        weights, bias = layer
        
        if (self.getNumberOfLayers()==0):
            self.network["weights"]["w1"] = weights
            self.network["biases"]["b1"] = bias
            self.network["activationFunctions"]["a1"] = self.getActivation(activation)
        else:
            num_layer = self.getNumberOfLayers()+1
    #        if (np.array(network["weights"]["w{}".format(num_layer-1)]).shape[0] == np.array(weights).shape[1]):
            self.network["weights"]["w{}".format(num_layer)] = weights
            self.network["biases"]["b{}".format(num_layer)] = bias
            self.network["activationFunctions"]["a{}".format(num_layer)] = self.getActivation(activation)
    #        else:
    #            raise ValueError()

    # Calculate neuron activation for an input
    def _activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    def forwardPass(self,x):
        a=x
        for i in range(1,self.getNumberOfLayers()+1):
            
    #        y = activate(network["weights"]["w{}".format(i)], a)
            
                    
            y = np.matmul(self.network["weights"]["w{}".format(i)], a) 

            if y.ndim < 2:
                y = np.expand_dims(y, axis=1)
            
            y += self.network["biases"]["b{}".format(i)]
            
            #get activation expression from dict
            act_function_name = self.network["activationFunctions"]["a{}".format(i)]
            act_function = self.activationFunctions[act_function_name]
            
            #apply function to y
            a=np.vectorize(act_function)(y)
            
    #        a = act_function(y)
            
            self.network["fp_activation"]["w{}".format(i)]  = a
            
        return a


    def getLastLayer(self):
        return self.network["weights"]["w{}".format(self.getNumberOfLayers())]


    def addLayer(self,numberOfNeuron,inputSize=None, activation="arctan"):

        #case of input layer    
        if (self.getNumberOfLayers() == 0 and inputSize != None):
            
            w1=createMatrix(numberOfNeuron,inputSize)
            b1=createMatrix(numberOfNeuron,1)
            
            self.addLayerToNetwork((w1,b1),activation)
            
        elif (self.getNumberOfLayers() == 0 and inputSize == None):
            raise ValueError("Expected inputSize")
        
        else:
            
            sizeOfLastLayer = np.array(self.getLastLayer()).shape[0]
            w=createMatrix(numberOfNeuron,sizeOfLastLayer)
            b=createMatrix(numberOfNeuron,1)
            
            self.addLayerToNetwork((w,b),activation)

    def _getDerivativeFromLayer(self, layer_index):
        
        act_function_name = self.network["activationFunctions"]["a{}".format(layer_index)]
                
        act_function_deriv = self.activationFunctions["d_"+act_function_name]
        
        return act_function_deriv

    def train(self, X, Y, LR=0.0001, iter_count=10000):
        """
        Implements the backpropagation algorithm to train the network
        """
        old_mean_error = 0
        num_layers = self.getNumberOfLayers()
        
        for _ in range(iter_count):
            
            mean_error = 0
            
            error = 0
            
            for index, x in enumerate(X):
                
                y = Y[index]
                
                if x.ndim < 2:
                    x = np.expand_dims(x, axis=1)
                
                for i in reversed(range(1,num_layers+1)):
                    
                    pred = self.forwardPass(x)
                    
                    z = self.network["fp_activation"]["w{}".format(i)]
                    act_function_deriv = self._getDerivativeFromLayer(i)
                    z_prime = np.vectorize(act_function_deriv)(z)            
                    if z_prime.ndim < 2:
                        z_prime = np.expand_dims(z_prime, axis=1)
                    
                    if (i==num_layers):
                                    
                        error = self.calcError(y, pred)
                        mean_error += error
                    
                    elif (i != num_layers):
                        
                        delta = self.network["deltas"]["w{}".format(i+1)]
                        w_i = self.network["weights"]["w{}".format(i+1)]
                        
                        error = 0.0
                        
                        for j in range(len(w_i)):
                            error += w_i[j] * delta[j]
                            
                        if error.ndim <2:                    
                            error = np.expand_dims(error, axis=1)

                    


                    delta = np.multiply(z_prime, error)
                    if delta.ndim < 2:
                        delta = np.expand_dims(delta, axis=1)
                    
                    self.network["deltas"]["w{}".format(i)] = delta
                
                for i in range(1,num_layers+1):
                    
                    #print("1--\t", network["deltas"]["w{}".format(i)].shape, a.shape)
                    
                    if i == 1:
                        a = x
                    else:
                        a = self.network["fp_activation"]["w{}".format(i-1)]
                    
                    product_for_update_w = np.matmul(self.network["deltas"]["w{}".format(i)], a.T)
                    product_for_update_b = self.network["deltas"]["w{}".format(i)]
            
                    if product_for_update_w.ndim < 2:
                        product_for_update_w = np.expand_dims(product_for_update_w, axis=1)
                        
                    #print(network["weights"]["w{}".format(i)].shape, product_for_update.shape)
                    
                    self.network["weights"]["w{}".format(i)] += LR * product_for_update_w
                    self.network["biases"]["b{}".format(i)] += LR * product_for_update_b
                    
                    
                    #print("2--\t",network["weights"]["w{}".format(i)].shape, a.shape)
                    
                    a = np.matmul(self.network["weights"]["w{}".format(i)], a)
                    
                    if a.ndim < 2:
                        a = np.expand_dims(a, axis=1)
                        
            mean_error = mean_error / len(X)

            # if abs(mean_error - old_mean_error) < 0.001:
            #     return mean_error
            
            old_mean_error = mean_error
            
        return old_mean_error
    
    def predict(self, X):
        res = []
        for x in X:

            if (hasattr(x, "__len__") == False):
                x = [x]

            y = np.asscalar(self.forwardPass(x))

            res.append(y)
        return res

    def test(self, X_test, y_test):
        error_res = 0
        for (x,y) in zip(X_test, y_test):

            if (hasattr(x, "__len__") == False):
                x = [x]

            y_pred = np.asscalar(self.forwardPass(np.array(x)))
            error_res += np.asscalar(self.calcError(y, [y_pred]))

        return error_res/y_test.shape[0]

    def train_with_pso(self, X_train, y_train, swarm_size=100, iter_count=10):
        pso = PSO(self, swarm_size)
        #for x, y in tqdm(zip(X_train, y_train)):
        network, perf = pso.train_nn(X_train, y_train, max_time=iter_count)
        self.network = network.network        
        return perf