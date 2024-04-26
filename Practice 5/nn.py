import numpy as np
from scipy.optimize import minimize

from computation import *
from data import *
from utils import *

def test_neuronal_network(X, y, theta1, theta2):
    m = len(y)
    y = np.ravel(y)
    
    _, _, result = forward_propagation(X, theta1, theta2)
    result = np.argmax(result, axis = 1)
    
    return (np.sum(result + 1 == y) / m * 100)

def initialize_weights(input_size, hidden_size, output_size):
    
    INIT_EPSILON = 0.12
    theta1 = np.random.random((hidden_size, (input_size + 1))) * (2 * INIT_EPSILON) - INIT_EPSILON
    theta2 = np.random.random((output_size, (hidden_size + 1))) * (2 * INIT_EPSILON) - INIT_EPSILON
    
    return theta1, theta2

def backprop(theta1, theta2, X, y, lambda_):
    
    # Cálculo del coste regularizado
    J = cost_reg(theta1, theta2, X, y, lambda_)
    
    # Gradientes regularizados
    gradient_1, gradient_2 = gradients(theta1, theta2, X, y)
    
    return J, gradient_1, gradient_2  

def train_neural_network(X, y, input_size, hidden_size, output_size, num_iterations, alpha, lambda_):
    
    # Obtención de los pesos
    theta1, theta2 = initialize_weights(input_size, hidden_size, output_size)
    
    for i in range(num_iterations):
        
        # Cálculo del coste y los diferentes gradientes
        cost, grad1, grad2 = backprop(theta1, theta2, X, y, lambda_)
        
        # Actualizar los parámetros con el descenso de gradiente
        theta1 -= alpha * grad1 
        theta2 -= alpha * grad2
        
        if i % 100 == 0:
            print(f'Iteración {i}: Costo = {cost}')
    
    return theta1, theta2

def learning_parameters():
    X, y = load_data()
    y_onehot = one_hot(y)
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    hidden_size = 25
    num_iterations = 1000
    alpha = 1
    lambda_ = 1
    
    theta1, theta2 = train_neural_network(X, y_onehot, input_size, hidden_size, output_size, num_iterations, alpha, lambda_)
    
    accuracy = test_neuronal_network(X, y, theta1, theta2)
    print(f'Precisión del entrenamiento: {accuracy}%')
    
def backprop_normalize(theta, input_size, hidden_size, labels_size, X, y, lambda_):
    theta1 = np.reshape(theta[:hidden_size * (input_size + 1 )], (hidden_size, (input_size + 1)))
    theta2 = np.reshape(theta[labels_size * (input_size + 1):], (labels_size, (hidden_size + 1)))
    
    m = len(y)
    
    J = cost_reg(theta1, theta2, X, y, lambda_)
    gradient_1, gradient_2 = gradients_reg(theta1, theta2, X, y, lambda_)
    
    gradient = np.concatenate((np.ravel(gradient_1)), (np.ravel(gradient_2)))
    
    return J, gradient
    
def learning_parameters_minimize():
    X, y = load_data()
    y_onehot = one_hot(y)
    input_size = X.shape[1]
    output_size = len(np.unique(y))
    hidden_size = 25
    num_iterations = np.arange(0, 200, 30)
    lambdas = [0.1, 1, 5, 10]
    
    theta1, theta2 = initialize_weights(input_size, hidden_size, output_size)
    theta = np.concatenate((np.ravel(theta1), np.ravel(theta2)))
    
    plt.figure()
    i = 0
    
    for lambda_ in lambdas:
        percent = []
        for iteration in num_iterations:
            fmin = minimize(
                fun = backprop_normalize, 
                x0 = theta,
                args = (input_size, hidden_size, output_size, X, y_onehot, lambda_),
                method = 'TNC', jac = True,
                options = { 'maxiter': iteration})
            print(fmin)
    return

def main():
    learning_parameters_minimize()

main()