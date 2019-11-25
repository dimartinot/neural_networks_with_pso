# neural_networks_with_pso
 Training a Neural Network with PSO and comparing its performance to homemade backpropagation algorithm.

# Main Code
The main python code and classes is situated in the "/classes" folder. Two classes are to distinguish:
 - Network
 - PSO

The network class implements both the feedfoward and the backpropagation algorithm (in the method train) for an ANN.
The PSO class implements the PSO algorithm, giving the user the choice of the hyperparameters and of the velocity update function.

In the "utils.py" file, you will find utility functions that are used throughout this project.

The main.py file is an obsolete file we used at first to run experiment. Due to its lack of flexibility, we changed for a Jupyter notebook.

# Experiments
The run experiments are found in the experiments.ipynb. This is a jupyter notebook and it can be run by typing the following command in a terminal:
 > jupyter notebook

This will open a new tab in the default browser and the experiments notebook will be choosable. Some experiments take multiple dozens of minutes to run unfortunately (on a i7, 16go ram machine). The results of these experiments, stored as Python dictionnaries, have been "*pickled*" into 3 files that can be loaded in code using:

```python
import pickle

variable = pickle.load( open(filename, "rb" ) )
```

where filename can be equal to "results_experiments_one.dictionnary", "results_experiments_two.dictionnary","results_experiments_three.dictionnary".