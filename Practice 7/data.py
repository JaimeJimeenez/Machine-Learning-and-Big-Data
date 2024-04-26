import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skl
from scipy.io import loadmat

def load_data(file):
    m = loadmat(file)
    X = m['X']
    y = m['y'].ravel()

    return X, y

def print_data(X, y):
    x0 = (X[:, 0].min(), X[:, 0].max())
    x1 = (X[:, 1].min(), X[:, 0].max())
    p = np.linspace(x0, x1, 100)

    x1, x2 = np.meshgrid(p, p)
    zeros = (y == 0).ravel()
    ones = (y == 1).ravel()

    plt.figure()
    plt.scatter(X[ones, 0], X[ones, 1], color='b', marker='x')
    plt.scatter(X[zeros, 0], X[zeros, 1], color='r', marker='o')

    plt.show()

def border_data(X, y, svm):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)
    plt.show()
    plt.close()