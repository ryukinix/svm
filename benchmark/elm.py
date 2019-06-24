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


def train(X, y, q=10, activation=None):
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
    # Utiliza-se mínimos quadrados
    M = D @ (Z.T @ (np.linalg.inv(Z @ Z.T)))

    return W, M


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

    def __init__(self, q=10, activation=processing.sigmoid):
        self.activation = activation
        self.q = q
        self.W = None
        self.M = None

    def fit(self, X, y):
        self.W, self.M = train(X, y, q=self.q, activation=self.activation)

    def predict(self, X):
        y_pred = predict(X, self.W, self.M, activation=self.activation)
        return processing.encode_label(y_pred)

    def score(self, X, y):
        y_encoded = processing.encode_label(y)
        y_pred = processing.encode_label(y_encoded)
        return testing.accuracy(y, y_pred)
