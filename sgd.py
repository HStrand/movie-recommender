import numpy as np
from matplotlib import pyplot
import pandas
from pandas import HDFStore


def sgd(R,Theta,X,K,iterations = 1000000, alpha=0.0005, beta=0.02):
    errors = []
    ploterrors = []
    X = X.T # Transpose movie feature matrix to be conformable in dot products
    for iter in range(iterations):
        if (iter+1)%100000 == 0:
            print("Starting iteration", iter+1)
            emean = np.mean(errors[iter-99999:]) # Hack to avoid indexing error at first 100k checkpoint
            print("Moving average error:", emean)
            ploterrors.append(emean)
        try:
            e = 0
            rnd = int(np.round((len(R)-1)*np.random.rand()))  # Select random user
            i = R['userId'][rnd]
            j = R['movieId'][rnd]
            fasit = R['rating'][rnd]

            eij = fasit - Theta[i,:].dot(X[:,j])  # Compute prediction error for sample
            Theta[i,:] +=  alpha*(2*eij*X[:,j] - beta*Theta[i,:])  # Adjust user features
            X[:,j] +=  alpha*(2*eij*Theta[i,:] - beta*X[:,j])  # Adjust movie features

            e += pow(fasit - Theta[i,:].dot(X[:,j]), 2)  # Compute squared error for plotting purposes
            errors.append(e)
        except MemoryError:
            print("memory error")

    return Theta, X.T, ploterrors


def learn(iterations=1000000):
    print("Loading HDF5 file...")
    store = HDFStore('ratings.h5')
    R = store['r100k']

    N = max(R['userId'])
    M = max(R['movieId'])
    K = 20

    print("Generating user features...")
    Theta = np.random.rand(N+1, K)
    print("Generating movie features...")
    X = np.random.rand(M+1, K)

    print("Starting Stochastic Gradient Descent...")
    nTheta, nX, errors = sgd(R, Theta, X, K, iterations)

    pyplot.figure(figsize=(10, 6))
    pyplot.plot(errors)
    pyplot.title("Loss function")
    pyplot.grid(True)
    pyplot.show()

    return nTheta, nX, errors


def predict(user, movie, Theta, X):
    X = X.T
    rating = Theta[user,:].dot(X[:,movie])
    print("Prediction for user", user, "movie", movie, ":", rating)
    return rating
