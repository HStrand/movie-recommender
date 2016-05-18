import numpy as np
import matplotlib
from matplotlib import pyplot
from pylab import *


def matrix_factorization(R, Theta, X, K, epochs=3000, alpha=0.0002, beta=0.02):
    trainingerrors = []
    X = X.T
    print(epochs, "epochs:")
    for epoch in range(epochs):
        if (epoch+1)%100 == 0:
            print("Starting epoch", epoch+1)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - Theta[i, :].dot(X[:, j])
                    for k in range(K):
                        Theta[i][k] += alpha * (2 * eij * X[k][j] - beta * Theta[i][k])
                        X[k][j] += alpha * (2 * eij * Theta[i][k] - beta * X[k][j])
        e = 0
        trainingcount = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    trainingcount += 1
                    e += pow(R[i][j] - Theta[i, :].dot(X[:, j]), 2)
                    for k in range(K):
                        e += (beta / 2) * (pow(Theta[i][k], 2) + pow(X[k][j], 2))
        trainingerrors.append(e / trainingcount)
        if e < 0.001:
            break
    return Theta, X.T, trainingerrors


def predict(Theta,X,R,user,movie):
    Z = X.T
    prediction = Theta[user,:].dot(Z[:,movie])
    realvalue = R[user,movie]
    print("Prediction: ", prediction)
    print("Real value: ", realvalue)
    print("Error: ", prediction-realvalue)


def SGD(R,Theta,X,K,iterations=1000000, alpha=0.0002, beta=0.02):
    errors = []
    X = X.T
    for iter in range(iterations):
        e = 0
        i = int(np.round(9*np.random.rand()))  # Select random user
        j = int(np.round(9*np.random.rand()))  # Select random movie
        if R[i][j] == 0: # Skip NA samples
            continue

        eij = R[i][j] - Theta[i,:].dot(X[:,j])  # Compute prediction error for sample
        Theta[i,:] = Theta[i,:] + alpha*(2*eij*X[:,j] - beta*Theta[i,:])  # Adjust user features
        X[:,j] = X[:,j] + alpha*(2*eij*Theta[i,:] - beta*X[:,j])  # Adjust movie features

        e += pow(R[i][j] - Theta[i,:].dot(X[:,j]), 2)  # Compute error again for plotting purposes
        errors.append(e)

    return Theta, X.T, errors


if __name__ == "__main__":
    R = [
        [5, 3, 0, 1, 0, 3, 5, 0, 2, 0],
        [4, 0, 0, 1, 5, 4, 0, 0, 2, 2],
        [1, 1, 0, 5, 1, 3, 5, 5, 0, 0],
        [1, 0, 0, 4, 0, 0, 0, 2, 3, 1],
        [0, 1, 5, 4, 4, 4, 2, 3, 0, 1],
        [0, 3, 2, 4, 0, 5, 0, 1, 1, 1],
        [4, 5, 2, 0, 2, 5, 0, 1, 0, 0],
        [5, 3, 0, 0, 3, 2, 1, 4, 0, 4],
        [0, 0, 0, 5, 0, 0, 5, 5, 0, 5],
        [0, 5, 3, 3, 0, 0, 0, 0, 3, 5]
    ]
    R = np.array(R)

    N = len(R)
    M = len(R[0])
    K = 5

    Theta = np.random.rand(N, K)
    X = np.random.rand(M, K)

    # nTheta, nX, errors = matrix_factorization(R, Theta, X, K)
    nTheta, nX, errors = SGD(R, Theta, X, K)
    nR = nTheta.dot(nX.T)

    error = 0
    count = 0
    for i in range(len(R)):
        for j in range(len(R[i])):
            if R[i, j] > 0:
                error += pow(R[i, j] - nR[i, j], 2)
                count += 1
    print("Finished. Mean squared error:", error/count)

    print("Example prediction:")
    predict(nTheta,nX,R,4,4)

    pyplot.figure(figsize=(10, 6))
    pyplot.plot(errors)
    pyplot.title("Loss function")
    pyplot.grid(True)
    pyplot.show()
