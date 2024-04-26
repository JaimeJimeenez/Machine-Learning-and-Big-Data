import numpy as np
import copy
import math
import matplotlib.pyplot as plt

from public_tests import *

def load_data_multi():
    data = np.loadtxt("data/houses.txt", delimiter=',')
    X = data[:,:4]
    y = data[:,4]
    return X, y


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.mean(X, axis=0) 
    sigma = np.std(X, axis=0)  
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def compute_cost(X, y, w, b):
    """
    Compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    m = len(y)
    predictions = np.dot(X, w) + b
    squarred_errors = np.square(predictions - y)
    cost = np.sum(squarred_errors) / (2 * m)
    return cost



def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m = len(y)
    predictions = np.dot(X, w) + b
    errors = predictions - y
    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.sum(errors) / m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function,
                     gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model
          after running gradient descent
      b : (scalar)                Updated value of parameter of the model 
          after running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history[i] = cost_function(X, y, w, b)

    return w, b, J_history

def main():
  X, y = load_data_multi()
  X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
  X_norm, mu, sigma = zscore_normalize_features(X)
  compute_cost_test(compute_cost)
  compute_gradient_test(compute_gradient)
  w, b, J_history = gradient_descent(X=X_norm, y=y, w_in=np.zeros(X_norm.shape[1]), b_in=0, cost_function=compute_cost, gradient_function=compute_gradient, alpha=0.01, num_iters=1000)
  
  house_features = np.array([1200, 3, 1, 40])
  house_features_normalized = (house_features - mu) / sigma
  predicted_price = np.dot(house_features_normalized, w) + b
  print("Predicted price for the house: $", predicted_price)
  
  X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
  fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
  for i in range(len(ax)):
    ax[i].scatter(X_norm[:, i], y)
    ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Price (1000's)")
    ax[i].scatter(X_norm[:, i], np.dot(X_norm, w) + b, color='orange')
  plt.show()
main()
