import numpy as np
import copy
import math

from public_tests import *
from utils import *

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    g = 1 / (1+ np.exp(-z))
    
    return g


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    m = len(y)

    z = np.dot(X, w) + b
    h = sigmoid(z)

    loss = -y * np.log(h) - (1 - y) * np.log(1 - h)
    total_cost = np.sum(loss) / m

    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """

    m = len(y)

    z = np.dot(X, w) + b
    h = sigmoid(z)

    dj_db = np.sum(h - y) / m
    dj_dw = np.dot(X.T, h - y) / m

    return dj_db, dj_dw

#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    m = len(y) 
    z = np.dot(X, w) + b
    h = sigmoid(z)

    loss = -y * np.log(h) - (1 - y) * np.log(1 - h)
    data_term = np.sum(loss) / m
    reg_term = (lambda_ / (2 * m)) * np.sum(w**2)
    total_cost = data_term + reg_term
    
    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    
    m = len(y) 
    z = np.dot(X, w) + b
    h = sigmoid(z)

    # Compute gradients
    dj_db = np.sum(h - y) / m
    dj_dw = np.dot(X.T, h - y) / m + (lambda_ / m) * w
    dj_dw[0] -= (lambda_ / m) * w[:]  # Exclude bias term from regularization

    return dj_db, dj_dw


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    w = w_in.copy()
    b = b_in
    J_history = []

    for _ in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        b -= alpha * dj_db
        w -= alpha * dj_dw
        cost = cost_function(X, y, w, b, lambda_)
        J_history.append(cost)

    return w, b, np.array(J_history)


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    z = np.dot(X, w) + b
    h = 1 / (1 + np.exp(-z))
    p = (h >= 0.5).astype(int)

    return p


def main():
  sigmoid_test(sigmoid)
  compute_cost_test(compute_cost)
  compute_gradient_test(compute_gradient)
  predict_test(predict)
  compute_cost_reg_test(compute_cost_reg)
  compute_gradient_reg_test(compute_gradient_reg)
main()

