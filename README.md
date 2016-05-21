# movie-recommender
Social movie recommender app.

The script sgd.py runs a matrix factorization algorithm that is optimized through Stochastic Gradient Descent.

## Quickstart

Open a Python (3.x) console and import the module with `from sgd import *`. Run experiments with the learn() function and predict movie ratings with the predict() function.


## Function learn()

Runs an experiment.

Arguments:

iterations (optional)

The learn() function collects a dataset from the HDF5 binary store ratings.h5. If no iteration argument is given, the default is set to one million. A moving average of the loss function will be printed out to the console every 100 000 iterations. Once the experiment is finished, the loss function is shown in a plot.

Example usage:
`Theta, X, e = learn()` runs an experiment.


## Function predict()

Predicts a movie rating.

Arguments:

user
movie
Theta
X

Theta and X are here the output from the learn function, where Theta is the matrix containing the learned user features and X is the matrix containing the learned movie features.

Example usage: 
`predict(1, 1, Theta, X)` predicts the rating user 1 will give to movie 1.


## Function sgd()

The core Stochastic Gradient Descent function, called by learn().

Arguments:

1. R
2. Theta
3. X
4. K
5. iterations
6. alpha (optional)
7. beta (optional)
