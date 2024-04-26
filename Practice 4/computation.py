import numpy as np
from public_tests import *

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, w, b, lambda_ = None):
    g = sigmoid(np.dot(X, w) + b)
    sum1 = np.dot(y, np.log(g))
    sum2 = np.dot((1 - y), np.log(1 - g))
    return (-1 / X.shape[0]) * (sum1 + sum2)

def gradient(X, y, w, b, lambda_ = None):
    m = X.shape[0]
    sigmoid_ =  sigmoid(np.dot(X, w) + b)
    dj_dw= (1 / m) *  np.dot(X.T, (sigmoid_ - y))
    dj_db = (1 / m) * np.sum((sigmoid_ - y))
    return dj_db, dj_dw

def cost_reg(X, y, w, b, lambda_ = 1):
    costs = cost(X, y, w, b, lambda_)
    m = X.shape[0]
    sum2 = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return costs + sum2

def gradient_reg(X, y, w, b, lambda_ = 1):
    m = X.shape[0]
    dj_db, dj_dw = gradient(X, y, w, b, lambda_)
    dj_dw += (lambda_ * w) / m
    return dj_db, dj_dw

def prepare_data(X, y, label):
    y_2 = (y == label) * 1
    y_2 = np.ravel(y_2)
    return (X, y_2)

def prepare_functions(lambda_):
    cost_ = lambda theta, X, y : cost_reg(X = X, y = y, b = theta, lambda_ = lambda_)
    gradient_ = lambda theta, X, y : gradient_reg(X = X, y = y,)
    return cost_, gradient_

def optimize_lambda(X, y, lambda_, label):
    X, y = prepare_data(X, y, label)
    cost_, gradient_ = prepare_functions(lambda_) 
    
    T = np.zeros(X.shape[1])
    
    return

sigmoid_test(sigmoid)
compute_cost_test(cost)
compute_gradient_test(gradient)
compute_cost_reg_test(cost_reg)
compute_gradient_reg_test(gradient_reg)