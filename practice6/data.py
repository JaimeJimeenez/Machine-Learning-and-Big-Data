import numpy as np
import matplotlib.pyplot as plt

def gen_data(m, seed = 1, scale = 0.7):
    """ generate a data set based on a x^2 with added noise """
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train ** 2 + c
    y_train = y_ideal + scale * y_ideal * (np.random.sample((m, )) - 0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal

def display_data():
    # Generate data
    x_train, y_train, x_ideal, y_ideal = gen_data(m = 64)

    # Display data
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, label='Data')
    plt.plot(x_ideal, y_ideal, color='red', linestyle='--', label='Ideal')
    plt.title('Conjunto de Datos Generado')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
