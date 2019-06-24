#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: <project>
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

from io import StringIO
import urllib.request as request
import pandas as pd


def digits():
    """Download digits dataset and return as file-like object.

    X: 64 features as gray-scale intensity of 8x8 picture
    y: {0..9} classes
    Samples: 5620

    Source: "https://datahub.io/machine-learning/optdigits#readme"
    """
    # Abrindo dataset MNIST
    digits_ref = "http://ufc.lerax.me/datasets/digits/digits.csv"
    print("[load] optical digits (8x8) dataset from {}".format(digits_ref))
    response = request.urlopen(digits_ref).read()
    data = StringIO(response.decode('utf-8'))
    df = pd.read_csv(data)
    # Separando atributos X e Y
    X = df.drop("class", axis=1).values
    y = df["class"].values

    return X, y


def mnist_train():
    pass


def mnist_test():
    pass
