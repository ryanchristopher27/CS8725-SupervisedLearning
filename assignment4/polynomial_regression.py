from math import exp
import math
import os
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
  
# Reference
    # https://towardsdatascience.com/polynomial-regression-gradient-descent-from-scratch-279db2936fe9

def main():
    # Height
    X = pd.DataFrame({"X1": [1.47, 1.5, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.7, 1.73, 1.75, 1.78, 1.8, 1.83]})
    # Weight
    Y = pd.DataFrame({"weight": [52.21, 53.12, 54.48, 55.84, 57.2, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.1, 69.92, 72.19, 74.46]})

    X = X.copy()
    X.loc[:, 'base'] = 1
    X.loc[:, 'X2'] = X["X1"]**2

    features = ['base', 'X1', 'X2']
    X = X[features]
    
    noise = 1

    # Perform polynomial regression
    # weights = polynomial_regression_matrix(X, Y, features)
    weights = polynomial_regression_iterative(X, Y, features, noise)

    Y_pred = evaluate_all_with_noise(weights, X, noise)

    # Plot results
    plot_polynomial_regression_line(X, Y, Y_pred)
    plot_polynomial_regression_scatter(X, Y, Y_pred)


def polynomial_regression_matrix(A :pd.DataFrame, Y :pd.DataFrame, features :[]) -> []:
    weights = [random.uniform(0.0, 1.0) for x in range(len(features))]

    A_T = A.T

    B_hat_denominator = np.dot(A_T, A)
    B_hat_numerator = np.dot(A_T, Y)

    B_hat = np.divide(B_hat_numerator, B_hat_denominator)
    # B_hat = np.dot((B_hat_numerator**(-1)), B_hat_denominator)

    print(B_hat)

# Polynomial Regression Function
def polynomial_regression_iterative(X :pd.DataFrame, Y :pd.DataFrame, features :[], noise :float) -> []:
    # Initialize bias randomly
    weights = [random.uniform(0.0, 10.0) for x in range(len(features))]
    # weights = [0 for x in range(len(features))]

    learning_rate = 0.001
    epochs = 200
    noise_parameter = noise

    errors = []

    # Loop over all epochs
    for epoch in range(epochs):
        y_prediction = evaluate_all_with_noise(weights, X, noise_parameter)

        errors.append(get_error(y_prediction, Y))

        derivatives = get_derivatives(X, Y, y_prediction, features)

        weights = update_weights(weights, derivatives, learning_rate)

    plot_error(errors, epochs)

    return weights

# Helper Functions
def evaluate_with_noise(weights :[], x :float, noise_size :float) -> float:
    b0 = weights[0]
    b1 = weights[1]
    b2 = weights[2]

    noise = random.random() * noise_size

    y = b0 + (b1 * x) + (b2 * (x**2)) + noise

    return y


def evaluate_all_with_noise(weights :[], X :pd.DataFrame, noise_size :float) -> []:
    b0 = weights[0]
    b1 = weights[1]
    b2 = weights[2]

    noise = [random.random() * noise_size for _ in range(len(X))]

    y = b0 + (b1 * X['X1']) + (b2 * X['X2']) + noise

    return y


def normalize(X :pd.DataFrame) -> pd.DataFrame:
         
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
         
        return X


def get_error(y_pred :np.array, y :pd.DataFrame) -> float:
    errors = y_pred - y['weight']
    squared_errors = errors ** 2

    mean_squared_error = sum(squared_errors) / len(errors)

    return mean_squared_error


def get_derivatives(X :pd.DataFrame, y :pd.DataFrame, y_pred :pd.DataFrame, features :[]) -> []:
        derivatives = []

        for i, feature in enumerate(features):
            # derivative = -2 * sum( X[feature] * (y['weight'] - y_pred))
            derivative = (-2 / len(y)) * sum( X[feature] * (y['weight'] - y_pred))
            derivatives.append(derivative)

        return derivatives


def update_weights(weights :[], derivatives :[], learning_rate :float) -> []:
    new_weights = []
    for i in range(len(weights)):
        w = weights[i] - (learning_rate * derivatives[i])
        new_weights.append(w)

    return new_weights


def plot_error(errors :[], epochs :int) -> None:
    x = np.arange(1, epochs + 1)
    plt.plot(x, errors)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error per Epoch')
    # plt.savefig('assignment3/plots/training_error.png')
    plt.savefig(os.path.join('assignment4/plots', "training_error.png"))
    plt.show()


def plot_polynomial_regression_scatter(X :pd.DataFrame, y :pd.DataFrame, y_pred :pd.DataFrame) -> None:
    plt.scatter(X['X1'], y_pred, color = 'red', label = 'Predictions')
    plt.scatter(X['X1'], y['weight'], color = 'blue', label = 'Original')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Polynomial Regression of Height vs. Weight')
    plt.savefig(os.path.join('assignment4/plots', "poly_regression_scatter.png"))
    plt.show()


def plot_polynomial_regression_line(X :pd.DataFrame, y :pd.DataFrame, y_pred :pd.DataFrame) -> None:
    plt.plot(X['X1'], y_pred, color = 'red', label = 'Predictions')
    plt.scatter(X['X1'], y['weight'], color = 'blue', label = 'Original')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Polynomial Regression of Height vs. Weight')
    plt.savefig(os.path.join('assignment4/plots', "poly_regression_line.png"))
    plt.show()

if __name__ == "__main__":
    main()