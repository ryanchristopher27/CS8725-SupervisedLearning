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
    
    noise = 0.1

    #y = b0 + b1 * x + b2 * X^2 + noise

    # Perform polynomial regression
    # weights = polynomial_regression_matrix(X, Y, features)
    weights = polynomial_regression_iterative(X, Y, features, noise)

    # X_test_normalized = normalize(X)
    Y_test_pred = evaluate_all_with_noise(weights, X, noise)

    # plt.figure(figsize=(10,5))
    plt.plot(X['X1'], Y_test_pred, color = 'red', label = 'Predictions')
    plt.scatter(X['X1'], Y['weight'], color = 'blue', label = 'Original')
    plt.show()

    # The accuracy
    # accuracy = get_accuracy(Y_test_pred, Y_test)
    # print(f"Test Accuracy = {accuracy}")

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
    epochs = 1000
    noise_parameter = noise

    errors = []
    accuracies = []

    # Loop over all epochs
    for epoch in range(epochs):
        y_prediction = evaluate_all_with_noise(weights, X, noise_parameter)

        derivatives = get_derivatives(X, Y, y_prediction, features)

        weights = update_weights(weights, derivatives, learning_rate)

    # plot_error(errors, epochs)
    # plot_accuracy(accuracies, epochs)

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
    errors = y_pred - y
    squared_errors = errors ** 2

    mean_squared_error = sum(squared_errors) / len(errors)

    return mean_squared_error

def get_accuracy(y_pred :np.array, y:pd.DataFrame) -> float:
    correct_predictions = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y.iloc[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_pred)
    return accuracy
    # print(f"Accuracy = {accuracy / len(y_pred)}")

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

def normalize(X :pd.DataFrame) -> pd.DataFrame:
    return X - X.mean()

def plot_error(errors :[], epochs :int) -> None:
    x = np.arange(1, epochs + 1)
    plt.plot(x, errors)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Mean Squared Error per Epoch')
    # plt.savefig('assignment3/plots/training_error.png')
    plt.savefig(os.path.join('assignment3/plots', "training_error.png"))
    plt.show()

def plot_accuracy(accuracies :[], epochs :int) -> None:
    x = np.arange(1, epochs + 1)
    plt.plot(x, accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    # plt.savefig('assignment3/plots/training_accuracy.png')
    plt.savefig(os.path.join('assignment3/plots', "training_accuracy.png"))
    plt.show()

def plot_confusion_matrix(y_pred :np.array, y :pd.DataFrame, label_1 :str, label_2 :str) -> None:
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels= [label_1, label_2])
    cm_display.plot()
    # plt.savefig('assignment3/plots/test_confusion_matrix.png')
    plt.savefig(os.path.join('assignment3/plots', "test_confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    main()