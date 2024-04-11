import numpy as np
import copy
import matplotlib.pyplot as plt
import math

from utils import load_data

#########################################################################
# Cost function
#
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    
    m = len(x)
    total_cost = 0

    total_cost = np.sum((w * x + b - y) ** 2) 

    total_cost /= (2 * m)
    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    m = len(x)
    dj_dw = 0
    dj_db = 0
    
    dj_dw = np.sum(w * x + b - y)
    dj_db = np.sum((w * x + b - y) * x)

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    J_history = np.zeros(num_iters)
    
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_db
        b -= alpha * dj_dw
        J_history[i] = cost_function(x, y, w, b)

    return w, b, J_history

def prediction(w, b, people): 
    return w * people + b

def main() -> None:
    x, y = load_data()
    w, b, J_history = gradient_descent(x, y, w_in=0, b_in=0, cost_function=compute_cost, gradient_function=compute_gradient, alpha=0.01, num_iters=1500)
    print(w, b, J_history)
    
    predicted = prediction(w, b, people=35000)
    print('Predicción con 35000 personas {}'.format(predicted))
    predicted = prediction(w, b, people=70000)
    print('Predicción con 70000 personas: {}'.format(predicted))
    y_fit = prediction(x, b, w)

    plt.scatter(x, y, label='Datos originales')
    plt.plot(y_fit, x, color='red', label='Ajuste lineal')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000')
    plt.title('Profits vs. Population per city')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

main()