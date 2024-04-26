from scipy.io import loadmat
import numpy as np

def load_data(file='./data/ex3data1.mat') -> tuple :
    data = loadmat(file, squeeze_me=True)
    X = data['X']
    y = data['y']
    return X, y

def load_weights(file='./data/ex3weights.mat'):
    weights = loadmat(file, squeeze_me=True)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    return theta1, theta2

def calculate_exponent(value : int) -> int:
    exponent : int = 0
    while value > 1:
        value >>= 1
        exponent += 1
    return exponent

def check_onehot(y, y_onehot):
    for i in np.random.randint(5000, size=10):
        print("{} = {}".format(y_onehot[i], y[i]))

def one_hot(y : list):
    m = len(y)
    y_onehot = np.zeros((m, 10))
    for value in y:
        value = calculate_exponent(value)
    
    for i in range(m):
        y_onehot[i][y[i]] = 1
    return y_onehot
