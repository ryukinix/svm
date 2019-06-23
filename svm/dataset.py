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


def mnist():
    """Download mnist dataset and return as file-like object."""
    # Abrindo dataset MNIST
    # REF: "https://datahub.io/machine-learning/optdigits#readme"
    mnist_ref = "http://ufc.lerax.me/datasets/mnist.csv"
    print("[load] mnist dataset from {}".format(mnist_ref))
    return StringIO(request.urlopen(mnist_ref).read().decode('utf-8'))
