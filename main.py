# imports
# https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Model import LogisticRegressionUsingGD
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from tools import preparation, loadfasttext
from sklearn.preprocessing import OneHotEncoder

def load_data():
    categories_and_split = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 929]
    categories = categories_and_split[0:2]
    raw_data_INIT = fetch_20newsgroups(subset='train', categories=categories)
    return raw_data_INIT


if __name__ == "__main__":
    path = r'fast_cbow_300D'
    # load the data from the file
    loadres = loadfasttext(path)
    prepres = preparation("input.txt", "output.txt")

    # vector
    Tx = max([len(sentence) for sentence in prepres[0]])
    X = np.zeros((prepres[2], Tx, loadres[1]))
    onehot_encoder = OneHotEncoder(sparse=False)
    Y = onehot_encoder.fit_transform(prepres[1])
    classes = Y.shape[1]
    Y = Y.reshape((prepres[2], 1, classes))
    for i in range(prepres[2]):
        for j in range(len(prepres[0][i])):
            X[i, j, :] = loadres[0].wv[prepres[0][i][j]]

    alpha = 0.01
    epsilon = 0.4
    ###################################################################"
    type_penalitty = 'RIDGE'
    # Logistic Regression from scratch using Gradient Descent
    model = LogisticRegressionUsingGD(alpha, epsilon)
    model.fit(X, Y, np.zeros(X), type_penalitty)
    #accuracy = model.accuracy(X_test, y_test.flatten())
    #parameters = model.w_
   # print("The accuracy of the {} model is {}".format(type_penalitty, accuracy))
   # print("The model parameters using Gradient descent")
   # print("\n")
    #print(parameters)