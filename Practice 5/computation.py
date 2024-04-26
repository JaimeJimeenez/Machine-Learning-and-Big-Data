import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_propagation(X, theta1, theta2):
    m = X.shape[0]
    X = np.hstack([np.ones([m, 1]), X])
    
    hidden = sigmoid(np.dot(X, theta1.T))
    hidden = np.hstack([np.ones([m, 1]), hidden])
    
    result = sigmoid(np.dot(hidden, theta2.T))
    return X, hidden, result

def cost(theta1, theta2, X, y):
    A1, A2, h = forward_propagation(X, theta1, theta2)
    
    first_sum = y * np.log(h)
    second_sum = (1 - y) * np.log(1 - h + 1e-6)
    
    return (-1 / X.shape[0]) * np.sum(first_sum + second_sum)

def cost_reg(theta1, theta2, X, y, lambda_):
    costs = cost(theta1, theta2, X, y)
    m = X.shape[0]
    result = np.sum(np.sum(theta1[:, 1:] ** 2)) + np.sum(np.sum(theta2[:, 1:] ** 2))

    return costs + (lambda_/(2*m)) * result
    
def gradients_reg(theta1, theta2, X, y, lambda_):
    m = len(y)
    delta_hidden, delta_out = gradients(theta1, theta2, X, y)
    
    delta_hidden[:, 1:] = delta_hidden[:, 1:] + (lambda_ / m) * theta1[:, 1:]
    delta_out[:, 1:] = delta_out[:, 1:] + (lambda_ / m) * theta2[:, 1:]
    
    return delta_hidden, delta_out 

def gradients(theta1, theta2, X, y):
    
    # Creación de los deltas respectivas con la forma de theta pero incializados a cero
    Delta_hidden = np.zeros(np.shape(theta1))
    Delta_out = np.zeros(np.shape(theta2))
    
    m = len(y)
    
    # Se realiza la propagación hacia delante
    # Obtención de los parámetros:
    # X_: matriz con todos los valores de entrada incluyendo el término de sesgo como primera columna
    # hidden: matriz con los valores de la capa de oculta incluyendo la columna donde se incluye el término de sesgo
    # h: matriz con los valoers de la capa de salida
    A1, A2, h = forward_propagation(X, theta1, theta2)
    
    for k in range(m):
        a1_k = A1[k, :]
        a2_k = A2[k, :]
        a3_k = h[k, :]
        y_k = y[k, :]
        
        delta_out = a3_k - y_k
        sigmoid_gradient = (a2_k * (1 - a2_k))
        delta_hidden = np.dot(theta2.T, delta_out) * sigmoid_gradient
        
        Delta_hidden = Delta_hidden + np.dot(delta_hidden[1:, np.newaxis], a1_k[np.newaxis, :])
        Delta_out = Delta_out + np.dot(delta_out[:, np.newaxis], a2_k[np.newaxis, :])
         
    return Delta_hidden / m, Delta_out / m