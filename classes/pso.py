import numpy as np

from random import seed
import random
from tqdm import tqdm

class PSO:

    numberOfInfant=20
    alpha=0.3 # proportion of velocity to be retained
    beta=0.2 # proportion of personal best to be retained
    gamma=0.3 # proportion of the informants’ best to be retained
    delta=0.2 # proportion of global best to be retained

    def __init__(self, network, swarmsize):
        super().__init__()
        self.network = network
        #dimension=2
        self.P={
        "location": {},
        "velocity": {},
        "fittestX": {},
        "informants": {},
        "informant_best":{}}
        self.swarmsize = swarmsize
    
    def val_to_act_fun(self, val):
        
        functions = self.network.activationFunctions
        sorted_functions = [function for function in functions if function[:2]!="d_"]

        if val < 0:
            val = 0
        if val > 1:
            val = 1
        
        for i, function in enumerate(sorted_functions):
            # with 9 functions, if val = 1/9, then we return the second function, if val = 4/9 we return the 5th and etc..
            if val >= i/len(sorted_functions) and val < (i+1)/len(sorted_functions):
                return function

        return "identity"

    def getNetworkFromParticule(self, particuleValue):
        compteur = 0
        network = self.network
        for i in range(1,network.getNumberOfLayers()+1):
            size = network.network["weights"]["w{}".format(i)].size
            
            b_size = network.network["biases"]["b{}".format(i)].size

            new_weights = np.array(particuleValue).flatten()[compteur:compteur+size]
            bias = np.array(particuleValue).flatten()[compteur+size:compteur+size+b_size]
            act_fun = np.array(particuleValue).flatten()[compteur+size+b_size:compteur+size+b_size+1]
            size+=b_size # bias
            #size+=1 # act function
            compteur+=size
            network.network["weights"]["w{}".format(i)] = new_weights.reshape(network.network["weights"]["w{}".format(i)].shape)
            network.network["biases"]["b{}".format(i)] = bias.reshape(network.network["biases"]["b{}".format(i)].shape)
            network.network["activationFunctions"]["a{}".format(i)] = self.val_to_act_fun(act_fun)

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

            f = self.network.network["activationFunctions"]["a{}".format(i)]

            functions = self.network.activationFunctions
            sorted_functions = [function for function in functions if function[:1]!="d_"]

            val = sorted_functions.index(f)/len(sorted_functions)

            location.append(val)


        return np.array(location)

    def assessFitness(self, particule, y_train, X_train, is_best=False):
    
        new_net = self.getNetworkFromParticule(particule)
        
        error = 0
        
        for x, y in zip(X_train, y_train):

            if (hasattr(x, "__len__") == False):
                x = [x]

            pred = np.asscalar(new_net.forwardPass(np.array(x)))
            error += self.network.calcError(y, pred)
        
        # if (is_best):
        #     print(error)
        return error/y_train.shape[0]

    def train_nn(self, X_train, y_train, max_time=10, threshold=0.001):
        dimension = self.getParticuleFromNetwork().size
        
        for i in range(self.swarmsize):
            self.P["location"]["l{}".format(i)]=np.random.rand(dimension,1)
            self.P["velocity"]["v{}".format(i)]=np.random.rand(dimension,1)
            randomChoice=np.random.choice(self.swarmsize,self.numberOfInfant,replace=False) #select random particule
            self.P["informants"]["l{}".format(i)]=randomChoice
        best=self.P["location"]["l0"] #initialise the best one one
        for i in range(self.swarmsize): #initialise the fitest one
            self.P["fittestX"]["f{}".format(i)]=self.P["location"]["l{}".format(i)]
                
        iter_count = max_time
        count_similar_fitness = 5
        last_fitness = 0
        best_perf_hist = []
        while ( count_similar_fitness > 0 and iter_count != 0):
            iter_count -=1
        #for time in tqdm(range(max_time)): #(time<10): #Best is the ideal solution or we have run out of time

            for i in range(self.swarmsize): #record the fittest known location of X so far
                if self.assessFitness(self.P["location"]["l{}".format(i)], y_train, X_train) < self.assessFitness(self.P["fittestX"]["f{}".format(i)], y_train, X_train):
                    self.P["fittestX"]["f{}".format(i)]=self.P["location"]["l{}".format(i)]
                    
            for i in range(self.swarmsize): #The fittest known location ~x+ that any of the informants of ~x have discovered so far
                randomChoice=np.random.choice(self.swarmsize,self.numberOfInfant,replace=False) #select random particule
                randomChoice=np.append(randomChoice,i) #add the particule we are working on

                indexes_informants=(self.P["informants"]["l{}".format(i)])

                first_index = self.P["informants"]["l{}".format(i)][0]
                best_informant = self.P["location"]["l{}".format(first_index)]
                for index in indexes_informants:

                    informant = self.P["location"]["l{}".format(index)]

                    if self.assessFitness(informant,y_train, X_train) < self.assessFitness(best_informant, y_train, X_train):
                        best_informant=self.P["location"]["l{}".format(index)]

                self.P["informant_best"]["i{}".format(i)]=best_informant
                    
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
                                c*(self.P["informant_best"]["i{}".format(i)][dim] - self.P["location"]["l{}".format(i)][dim]) +
                                d*(best[dim] - self.P["location"]["l{}".format(i)][dim]))

                self.P["velocity"]["v{}".format(i)]=velo
                
            for i in range(self.swarmsize):
                newlocation=[]
                for dim in range(dimension):
                    newlocation.append(self.P["location"]["l{}".format(i)][dim] + self.P["velocity"]["v{}".format(i)][dim])
                self.P["location"]["l{}".format(i)] = newlocation
            #time=time+1
        
            #print(self.P)
            tmp_last_fitness = self.assessFitness(best, y_train, X_train, True)
            if (last_fitness == tmp_last_fitness):
                count_similar_fitness -=0
            else:
                count_similar_fitness = 5
                last_fitness = tmp_last_fitness
            
            best_perf_hist.append(last_fitness)
        
        print(f"Best_fitness: {last_fitness}")

        return self.getNetworkFromParticule(best), best_perf_hist