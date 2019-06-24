#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neural RBF
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""-- RBF Neural Network

Um algoritmo baseado em redes neurais e mínimos quadrados.
Os pesos da camada oculta são unitários.
Os neurônios da camada oculta são especiais: RBF -> Radial Base Function.
Pontos aleatórios do treinamento são assumidos como centróides da camada oculta.
"""


import numpy as np
import processing
from sklearn import base
import testing

Q = 10
TEST_SIZE = 0.2


def phi(X, T):
    """Neurônio RBF da camada oculta"""
    n, _ = X.shape
    q, p = T.shape
    matrix = np.zeros(shape=(n, q))
    for i in range(n):
        x = X[i]
        for j in range(q):
            t = T[j]
            matrix[i][j] = np.exp(np.linalg.norm(x - t))

    return matrix


def train(X, y, q=Q, activation=None):
    """Algoritmo de treinamento para Rede Neural RBF

    Parâmetros
    ---------
    X: Vetor de características
    y: Vetor de rótulos
    q: número de neurônios ocultos

    Return
    ------
    T: Centróides da camada oculta com tamanho q
    G: Matriz de pesos da camada de saída.


    """
    # rótulos
    n, p = X.shape
    index = np.arange(0, n - 1)
    D = y

    # Adicionar bias
    X = processing.add_bias(X)

    # training
    # Centróides aleatórios da camada oculta
    T = X[np.random.choice(index, q)]

    # Calcular saída da camada oculta
    PHI = phi(X, T)
    if activation is not None:
        PHI = activation(PHI)
    PHI = processing.add_bias(PHI, axis=1)

    # Calcular pesos G para camada de saída (aprendizado)
    # G = D @ (PHI.T @ (np.linalg.inv(PHI @ PHI.T)))
    G, *_ = np.linalg.lstsq(PHI, D, rcond=None)

    return T, G.T


def predict(X, T, G, activation=None):
    """Algoritmo de predição para ELM (Extreme Learning Machine)

    Parâmetros
    ----------
    T: Centróides da camada oculta
    G: Vetor de pesos da camada de saída para

    Return
    ------
    np.array

    """
    X = processing.add_bias(X)
    PHI = phi(X, T)
    if activation is not None:
        PHI = activation(PHI)
    PHI = processing.add_bias(PHI, axis=1)
    Y = G @ PHI.T

    return Y.T


class NNRBF(base.BaseEstimator):

    def __init__(self, q=Q, activation=processing.sigmoid, one_hot_y=True):
        self.activation = activation
        self.q = q
        self.T = None
        self.G = None
        self.one_hot_y = one_hot_y

    def fit(self, X, y):
        if self.one_hot_y:
            y = processing.one_hot_encoding(y)
        self.T, self.G = train(X, y, q=self.q, activation=self.activation)
        return self

    def predict(self, X):
        y_pred = predict(X, self.T, self.G, activation=self.activation)
        encoded = processing.encode_label(y_pred).astype(int)
        return encoded

    def score(self, X, y):
        y_pred = self.predict(X)
        y_encoded = processing.encode_label(y_pred)
        return testing.accuracy(y, y_encoded)


def main():
    import dataset
    X, y = dataset.digits()
    X_train, X_test, y_train, y_test = testing.hold_out(X, y)
    clf = NNRBF(q=1000)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
