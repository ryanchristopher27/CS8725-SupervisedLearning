from math import exp
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
  

def main():
    # fetch dataset 
    iris = fetch_ucirepo(id=53) 
    
    # Grab only Setosa and Versicolour
    X = iris.data.features[:100]
    y = iris.data.targets [:100]
    # Set Setosa to 0 and Versicolor to 1
    y = y['class'].replace({'Iris-setosa': 0, 'Iris-versicolor': 1})

    # Grab Versicolor and Virginica
    # X = iris.data.features[-100:]
    # y = iris.data.targets [-100:]
    # y = y['class'].replace({'Iris-versicolor': 0, 'Iris-virginica': 1})

    X['base'] = [1 for x in range(len(X.index))]

    features = ['base', 'sepal length', 'sepal width', 'petal length', 'petal width']

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 0 )
    

    weights = logistic_regression(X_train, Y_train, features)

    X_test_normalized = normalize(X_test)

    Y_test_pred = sigmoid(X_test_normalized, weights, features)

    Y_test_pred = [1 if p >= 0.5 else 0 for p in Y_test_pred]

    # plt.clf()
    # plt.scatter(X_test, Y_test)
    # plt.scatter(X_test, Y_test_pred, c="red")
    # plt.show()

    # The accuracy
    accuracy = 0
    for i in range(len(Y_test_pred)):
        if Y_test_pred[i] == Y_test.iloc[i]:
            accuracy += 1
    print(f"Accuracy = {accuracy / len(Y_test_pred)}")




def logistic_regression(X :pd.DataFrame, y :pd.DataFrame, features :[]) -> []:
    # features = ['base', 'sepal length', 'sepal width', 'petal length', 'petal width']

    # Initialize bias randomly
    weights = [random.uniform(0.0, 1.0) for x in range(len(features))]
    # weights = [0 for x in range(len(features))]

    learning_rate = 0.001
    epochs = 300

    X_normalized = normalize(X)

    # Loop over all epochs
    for epoch in range(epochs):
        y_prediction = sigmoid(X_normalized, weights, features)

        derivatives = get_derivatives(X_normalized, y, y_prediction, features)

        weights = update_weights(weights, derivatives, learning_rate)

    return weights

# Helper Functions
def sigmoid(X :pd.DataFrame, weights :[], features :[]) -> np.array:

    # z = sum([weights[i] * X[feature] for i, feature in enumerate(features)])

    predictions = []

    for i, x in X.iterrows():
        z = sum([weights[i] * x[feature] for i, feature in enumerate(features)])
        predictions.append(1 / (1 + exp(z)))
        # predictions.append(1 / (1 + exp(-sum([weights[i] * x[feature] for i, feature in enumerate(features)]))))

    return np.array(predictions)
    # return np.array([1 / (1 + exp(-sum([weights[i] * x[feature] for i, feature in enumerate(features)]))) for x in X])


def get_derivatives(X :pd.DataFrame, y :pd.DataFrame, y_pred: np.array, features :[]) -> []:
        derivatives = []
        for i, feature in enumerate(features):
            derivative = -2 * sum( X[feature] * (y - y_pred) * y_pred * (1 - y_pred))
            derivatives.append(derivative)

        return derivatives

def update_weights(weights :[], derivatives :[], learning_rate :float) -> []:
    new_weights = []
    for i in range(len(weights)):
        w = weights[i] + (learning_rate * derivatives[i])
        new_weights.append(w)

    return new_weights

def normalize(X :pd.DataFrame) -> pd.DataFrame:
    return X - X.mean()

if __name__ == "__main__":
    main()