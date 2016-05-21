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


def compute_user_averages(R):
    num = max(R['userId'])+1
    avg_user_ratings = [0]*num
    for i in range(num):
        avg_user_ratings[i] = (np.mean(R['rating'][R['userId']==i]))
    return avg_user_ratings


def compute_movie_averages(R):
    num = max(R['movieId'])+1
    avg_movie_ratings = [0]*num
    for i in range(num):
        avg_movie_ratings[i] = (np.mean(R['rating'][R['movieId']==i]))
    return avg_movie_ratings


def remove_nans(list):
    for i in range(len(list)):
        if(str(list[i])=='nan'):
            list[i] = 0

def normalize(R, movie_avgs, usereffects):
    pandas.options.mode.chained_assignment = None # Allow assignment back to R
    normalized = []
    for i in range(len(R)):
        R['rating'][i] -= (movie_avgs[0][R['movieId'][i]] + usereffects[0][R['userId'][i]])


def predict_normalized(user, movie, Theta, X, movie_avg, usereffect):
    X = X.T
    rating = movie_avg + usereffect + Theta[user,:].dot(X[:,movie])
    print("Prediction for user", user, "movie", movie, ":", rating)
    return rating
