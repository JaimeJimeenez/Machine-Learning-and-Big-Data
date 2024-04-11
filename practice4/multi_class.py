import numpy as np
from scipy.optimize import minimize

from computation import *
from utils import *

#########################################################################
# one-vs-all
#
def oneVsAll(X, y, n_labels, lambda_):
    m, n = X.shape
    all_theta = np.zeros(n_labels, n + 1)
    X = np.column_stack((np.ones((m, 1)), X))
    
    for c in range(n_labels):
        initial_theta = np.zeros(n + 1)
        y_one_vs_all = (y == c).astype(int)
        theta = minimize(cost, initial_theta, args=(X, y_one_vs_all, lambda_), method=None, jac=gradient, options={'maxiter':50}).x
        all_theta[c] = theta
    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """

    return p


#########################################################################
# NN
#
def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    return p

def main() :
    X, y = load_data()
    print(X)
    print(X.shape)
    oneVsAll(X, y, 10, 0.1)
    return

main()

