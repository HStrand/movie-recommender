# Social movie recommender app

The script sgd.py runs a collaborative filtering algorithm that is optimized through Stochastic Gradient Descent.

### Quickstart

Open a Python (3.x) console and import the module with `from sgd import *`. Run experiments with the learn() function and predict movie ratings with the predict() function.



### learn()

Runs an experiment.

**Arguments**

1. iterations (optional) - number of iterations, default is set to 1000000

The learn() function collects a dataset from the HDF5 binary store ratings.h5. A moving average of the loss function will be printed out to the console every 100 000 iterations. Once the experiment is finished, the loss function is shown in a plot.

**Example usage**

`Theta, X, errors = learn()` runs an experiment and stores the learned features and loss function.



### predict()

Predicts a movie rating.

**Arguments**

1. user

2. movie

3. Theta

4. X

Theta and X are here the output from the learn function, where Theta is the matrix containing the learned user features and X is the matrix containing the learned movie features.

**Example usage**

`predict(1, 1, Theta, X)` predicts the rating user 1 will give to movie 1.



### sgd()

The core Stochastic Gradient Descent function, called by learn().

**Arguments**

1. R - a ratings matrix of type pandas.DataFrame with the columns 'userId', 'movieId' and 'rating'.

2. Theta - a matrix containing randomly initialized user features

3. X - a matrix containing randomly initialized movie features

4. K - number of features

5. iterations (optional) - number of iterations, default is set to 1000000

6. alpha (optional) - learning rate, default is set to 0.0005

7. beta (optional) - regularization parameter, default is set to 0.02


**Example usage**

`Theta, X, errors = sgd(R, Theta, X, 20)`

