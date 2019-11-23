import numpy as np

def unknownDerivative():
    raise ValueError("?")

def l1_error(y_true,y_pred):
    return np.array(np.abs(y_true - y_pred))

def l2_error(y_true,y_pred):
    return (np.square(y_true - y_pred)).mean(axis=0)

def createMatrix(output_size, input_size):
    w=np.random.normal(0,1,(output_size,input_size))
#    w = np.ones((output_size,input_size))
#    w = [[random() for i in range(input_size + 1)] for i in range(output_size)]
    return w