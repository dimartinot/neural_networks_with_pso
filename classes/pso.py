import numpy as np

from random import seed
import random
from tqdm import tqdm

class PSO:

    numberOfInfant=10
    alpha=0.2 # proportion of velocity to be retained
    beta=0.2 # proportion of personal best to be retained
    gamma=0.2 # proportion of the informants’ best to be retained
    delta=0.1 # proportion of global best to be retained
    epsilon=1 # jump size of a particle

    def __init__(self, network, swarmsize):
        super().__init__()
        self.network = network
        #dimension=2
        self.P={
        "location": {},
        "velocity": {},
        "fittestX": {},
        "informants": {}}
        self.swarmsize = swarmsize

    def getNetworkFromParticule(self, particuleValue):
        compteur = 0
        network = self.network
        for i in range(1,network.getNumberOfLayers()+1):
            size = network.network["weights"]["w{}".format(i)].size
            
            b_size = network.network["biases"]["b{}".format(i)].size

            new_weights = np.array(particuleValue).flatten()[compteur:compteur+size]
            bias = np.array(particuleValue).flatten()[compteur+size:compteur+size+b_size]
            #act_fun = np.array(particuleValue).flatten()[compteur+size+1:compteur+size+2]
            size+=1 # bias
            #size+=1 # act function
            compteur+=size
            network.network["weights"]["w{}".format(i)] = new_weights.reshape(network.network["weights"]["w{}".format(i)].shape)
            network.network["biases"]["b{}".format(i)] = bias.reshape(network.network["biases"]["b{}".format(i)].shape)
        return network

    def getParticuleFromNetwork(self):
        location = []
        for i in range(1,self.network.getNumberOfLayers()+1):

            weights = self.network.network["weights"]["w{}".format(i)]
            for w in weights.flatten():
                location.append(np.array([w]))

            biases = self.network.network["biases"]["b{}".format(i)]
            for b in biases.flatten():
                location.append(np.array([b]))

        return np.array(location)

    def assessFitness(self, particule, y_train, X_train, is_best=False):
    
        new_net = self.getNetworkFromParticule(particule)
        
        error = 0
        
        for x, y in zip(X_train, y_train):

            if (hasattr(x, "__len__") == False):
                x = [x]

            pred = new_net.forwardPass(np.array(x))
            error += self.network.calcError(y, pred)
        
        # if (is_best):
        #     print(error)
        return error

    def train_nn(self, X_train, y_train, max_time=10):
        dimension = self.getParticuleFromNetwork().size
        
        for i in range(self.swarmsize):
            self.P["location"]["l{}".format(i)]=np.random.rand(dimension,1)
            self.P["velocity"]["v{}".format(i)]=np.random.rand(dimension,1)
        best=self.P["location"]["l0"] #initialise the best one one
        for i in range(self.swarmsize): #initialise the fitest one
            self.P["fittestX"]["f{}".format(i)]=self.P["location"]["l{}".format(i)]
                
        for time in tqdm(range(max_time)): #(time<10): #Best is the ideal solution or we have run out of time

    #        print(time)
            for i in range(self.swarmsize): #record the fittest known location of X so far
                if self.assessFitness(self.P["location"]["l{}".format(i)], y_train, X_train) < self.assessFitness(self.P["fittestX"]["f{}".format(i)], y_train, X_train, True):
                    self.P["fittestX"]["f{}".format(i)]=self.P["location"]["l{}".format(i)]
                    
            for i in range(self.swarmsize): #The fittest known location ~x+ that any of the informants of ~x have discovered so far
                randomChoice=np.random.choice(self.swarmsize,self.numberOfInfant,replace=False) #select random particule
                randomChoice=np.append(randomChoice,i) #add the particule we are working on

                informantsBest=(self.P["location"]["l{}".format(i)])
                for partRandom in randomChoice:
                    if self.assessFitness(self.P["location"]["l{}".format(partRandom)], y_train, X_train) < self.assessFitness(informantsBest, y_train, X_train):
                        informantsBest=self.P["location"]["l{}".format(partRandom)]
                self.P["informants"]["i{}".format(i)]=informantsBest
                    
            for i in range(self.swarmsize): #record the best particule ever
                if self.assessFitness(self.P["location"]["l{}".format(i)], y_train, X_train) < self.assessFitness(best, y_train, X_train): 
                    best=self.P["location"]["l{}".format(i)]
                    
            for i in range (self.swarmsize):
                velo=[]
                for dim in range (dimension):
                    b=random.uniform(0,self.beta)
                    c=random.uniform(0,self.gamma)
                    d=random.uniform(0,self.delta)
                    velo.append(self.alpha*self.P["velocity"]["v{}".format(i)][dim] + 
                                b*(self.P["fittestX"]["f{}".format(i)][dim] - self.P["location"]["l{}".format(i)][dim]) +
                                c*(self.P["informants"]["i{}".format(i)][dim] - self.P["location"]["l{}".format(i)][dim]) +
                                d*(best - self.P["location"]["l{}".format(i)][dim]))
                    
                self.P["velocity"]["v{}".format(i)]=velo[0]
                
            for i in range(self.swarmsize):
                newlocation=[]
                for dim in range(dimension):
                    newlocation.append(self.P["location"]["l{}".format(i)][dim] + self.P["velocity"]["v{}".format(i)][dim])
                self.P["location"]["l{}".format(i)] = newlocation
            #time=time+1
        
        return self.P["fittestX"]