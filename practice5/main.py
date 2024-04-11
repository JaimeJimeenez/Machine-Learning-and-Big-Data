import numpy as np
from scipy.optimize import minimize

from data import *
from utils import *

def sigmoid(z):
    """Función de activación sigmoide."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivada de la función de activación sigmoide."""
    return sigmoid(z) * (1 - sigmoid(z))

def forward_propagation(X, Theta1, Theta2):
    """Propagación hacia adelante para calcular las activaciones de las capas ocultas y de salida."""
    # Capa oculta
    a1 = np.insert(X, 0, values=1, axis=1)  # Agrega el término de sesgo
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    # Capa de salida
    a2 = np.insert(a2, 0, values=1, axis=1)  # Agrega el término de sesgo
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    return a1, z2, a2, z3, a3

def compute_cost(X, y, Theta1, Theta2, lambd=0):
    """Calcula la función de costo (regresión logística)."""
    m = len(X)
    _, _, _, _, h = forward_propagation(X, Theta1, Theta2)
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_term = (lambd / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))
    return cost + reg_term

def backpropagation(X, y, Theta1, Theta2, lambda_=0):
    """Propagación hacia atrás para calcular los gradientes."""
    m = len(X)
    delta1 = np.zeros_like(Theta1)
    delta2 = np.zeros_like(Theta2)

    a1, z2, a2, z3, h = forward_propagation(X, Theta1, Theta2)

    # Calcular el error en la capa de salida
    d3 = h - y

    # Calcular el error en la capa oculta
    d2 = np.dot(d3, Theta2[:, 1:]) * sigmoid_derivative(z2)

    # Acumular los gradientes
    delta1 += np.dot(d2.T, a1)
    delta2 += np.dot(d3.T, a2)

    # Regularización (omitimos la regularización para el término de sesgo)
    delta1[:, 1:] += (lambda_ / m) * Theta1[:, 1:]
    delta2[:, 1:] += (lambda_ / m) * Theta2[:, 1:]

    # Gradientes sin promedio
    D1 = (1 / m) * delta1
    D2 = (1 / m) * delta2

    return D1, D2

def checkGradients():
    X, y = load_data()
    y_onehot = one_hot(y)
    theta1, theta2 = load_weights()
    J, gradient_1, gradient_2 = backpropagation(theta1, theta2, X, y_onehot, lambda_= 0.1)
    checkNNGradients(backpropagation)
    return

# Ejemplo de uso:
# Definir los datos de entrada X, las etiquetas y, y los parámetros Theta1 y Theta2
# Llamar a backpropagation para calcular los gradientes
# Actualizar los parámetros Theta utilizando los gradientes y un algoritmo de optimización (por ejemplo, descenso de gradiente)
def main():
    checkGradients()
    return

main()