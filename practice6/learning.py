import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from data import *

def error(predictions, labels):
    m = len(labels)
    mse = np.sum((predictions - labels) ** 2) / (2 * m)
    return mse

def overfitting_example(degree = 15):
    x_train, y_train, x_ideal, y_ideal = gen_data(m = 64)
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=1)
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
    X_test_poly = poly.transform(X_test[:, np.newaxis])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    mse_train = error(y_train_pred, y_train)
    mse_test = error(y_test_pred, y_test)
    
    print("Error on training set:", mse_train)
    print("Error on test set:", mse_test)

    plt.scatter(X_train, y_train, label='train')
    plt.scatter(X_train, y_train_pred, label='predicted')
    plt.plot(x_ideal, y_ideal, label='y_ideal', color='orange')
    plt.legend()
    plt.show()
    
def select_polynomial_degree():
    x_train, y_train, _, _ = gen_data(64)
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

    degrees = range(1, 11)
    best_degree = None
    min_val_error = float('inf')

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
        X_val_poly = poly.transform(X_val[:, np.newaxis])

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        y_val_pred = model.predict(X_val_poly)
        val_error = error(y_val_pred, y_val)

        if val_error < min_val_error:
            min_val_error = val_error
            best_degree = degree

    print("Best polynomial degree:", best_degree)
    return best_degree

def select_lambda(degree = 15):
    x_train, y_train, _, _ = gen_data(64)
    X_train, X_val_test, y_train, y_val_test = train_test_split(x_train, y_train, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)

    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]
    best_lambda = None
    min_test_error = float('inf')

    for lambda_ in lambdas:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
        X_val_poly = poly.transform(X_val[:, np.newaxis])

        model = Ridge(alpha=lambda_)
        model.fit(X_train_poly, y_train)

        y_val_pred = model.predict(X_val_poly)
        test_error = error(y_val_pred, y_val)

        if test_error < min_test_error:
            min_test_error = test_error
            best_lambda = lambda_
            
    print("Best lambda:", best_lambda)
    return best_lambda

def hyperparameter_tuning():
    x_train, y_train, _, _ = gen_data(750)
    X_train, X_val_test, y_train, y_val_test = train_test_split(x_train, y_train, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)

    degrees = range(1, 16)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]
    best_degree = None
    best_lambda = None
    min_test_error = float('inf')

    for degree in degrees:
        for lmbda in lambdas:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
            X_val_poly = poly.transform(X_val[:, np.newaxis])

            model = Ridge(alpha=lmbda)
            model.fit(X_train_poly, y_train)

            y_val_pred = model.predict(X_val_poly)
            test_error = error(y_val_pred, y_val)

            if test_error < min_test_error:
                min_test_error = test_error
                best_degree = degree
                best_lambda = lmbda

    print("Best polynomial degree:", best_degree)
    print("Best lambda:", best_lambda)
    print('Error', min_test_error)
    return best_degree, best_lambda
    
def learning_curves():
    train_sizes = np.arange(50, 1001, 50)
    train_errors = []
    val_errors = []

    for size in train_sizes:
        x_train, y_train, _, _ = gen_data(size)
        X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

        poly = PolynomialFeatures(degree=16, include_bias=False)
        X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
        X_val_poly = poly.transform(X_val[:, np.newaxis])

        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        y_train_pred = model.predict(X_train_poly)
        y_val_pred = model.predict(X_val_poly)

        train_error = error(y_train_pred, y_train,)
        val_error = error(y_val_pred, y_val)

        train_errors.append(train_error)
        val_errors.append(val_error)

    plt.plot(train_sizes, train_errors, label='train error')
    plt.plot(train_sizes, val_errors, label='cv error')
    plt.xlabel('Number of Examples (m)')
    plt.ylabel('error')
    plt.legend()
    plt.show()

def main():
    best_degree = select_polynomial_degree()
    best_lambda = select_lambda()
    best_degree_hyper, best_lambda_hyper = hyperparameter_tuning()
    learning_curves()
main()


