from sklearn.datasets import make_blobs
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def display_data(X_train, X_test, y_train, y_test):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training data')
    axs[0, 0].set_title('Dataset (Training)')
    axs[0, 0].legend()

    axs[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='Testing data')
    axs[0, 1].set_title('Dataset (Testing)')
    axs[0, 1].legend()

    axs[1, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training data')
    axs[1, 0].set_title('Dataset (Training)')
    axs[1, 0].legend()

    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='Testing data')
    axs[1, 1].set_title('Dataset (Testing)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

def generate_data(): 
    classes = 6
    m = 800
    std = 0.4
    centers = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [-2, 1], [-2, -1]])
    X, y = make_blobs(n_samples=m, centers=centers, cluster_std=std,
    random_state=2, n_features=2)

    return X, y

def split_data(X, y, test_size):
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X, y_train, y_

def display_accuracy_loss(accuracy_hist, loss_hist):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_hist, lw = 3)
    ax.set_title('Training loss', size = 15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy_hist, lw = 3)
    ax.set_title('Training accuracy', size = 15)
    ax.set_xlabel('Epoch', size = 15)
    ax.tick_params(axis = 'both', which='major', labelsize=15)
    plt.tight_layout()

    plt.show()