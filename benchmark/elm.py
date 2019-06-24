#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Inteligência Computacional UFC 2019.1 - Redes Neurais
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

"""-- Extreme Learning Machine

Um algoritmo baseado em redes neurais e mínimos quadrados.
Os pesos são aleatórios com uma distribuição normal.

Os tipos de teste são:

+ Hold-Out (splt train test)
+ K-Fold (cross-validation)
+ Leave-One-Out (cross-validation com k=n)

1. Camada oculta com pesos aleatórios de Z = W * X

q = 10 -> 0.89 acc
q = 20 -> 0.97 acc (sorte do cassete)

"""


import numpy as np
import processing
from sklearn import base
import testing

Q = 10

def train(X, y, q=Q, activation=None):
    """Algoritmo de treinamento para ELM (Extreme Learning Machine)

    Parâmetros
    ---------
    X: Vetor de características
    y: Vetor de rótulos
    q: número de neurônios ocultos

    Return
    ------
    W: pesos aleatórios da camada oculta

    """
    # rótulos
    # torna vetor linha em coluna
    n, p = X.shape
    D = y.T

    # training
    # Pesos aleatórios da camada oculta
    W = np.random.randn(q, p+1)
    # Adicionar bias
    X = processing.add_bias(X)
    # Calcular saída da camada oculta
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = processing.add_bias(Z, axis=0)
    # Calcular pesos M para camada de saída (aprendizado)
    # M = D @ (Z.T @ (np.linalg.inv(Z @ Z.T)))
    M, *_ = np.linalg.lstsq(Z.T, D.T, rcond=None)

    return W, M.T


def predict(X, W, M, activation=None):
    """Algoritmo de predição para ELM (Extreme Learning Machine)

    Parâmetros
    ----------
    W: Vetor de pesos da camada oculta utilizados no treinamento
    M: Vetor de pesos da camada de saída para

    Return
    ------
    np.array

    """
    X = processing.add_bias(X)
    Z = W @ X.T
    if activation is not None:
        Z = activation(Z)
    Z = processing.add_bias(Z, axis=0)
    Y = M @ Z

    return Y.T


class ELM(base.BaseEstimator):

    def __init__(self, q=Q, activation=processing.sigmoid, one_hot_y=True):
        self.activation = activation
        self.q = q
        self.W = None
        self.M = None
        self.one_hot_y = one_hot_y

    def fit(self, X, y):
        if self.one_hot_y:
            y = processing.one_hot_encoding(y)
        self.W, self.M = train(X, y, q=self.q, activation=self.activation)
        return self

    def predict(self, X):
        y_pred = predict(X, self.W, self.M, activation=self.activation)
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
    clf = ELM(q=100)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main()
