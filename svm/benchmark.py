# coding: utf-8

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt

import dataset
from elm import ELM

plt.style.use('grayscale')


def plot(names, results):
    # Construção gráfica da box, contendo o comparativo entre os aloritimos
    fig = plt.figure()
    fig.suptitle('Comparação dos algoritimos')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.xlabel("Modelos")
    plt.ylabel("Acurácia")
    plt.show()


def main():
    X, y = dataset.digits()
    print("--X--")
    print(X)
    print("SHAPE: ", X.shape)
    print("--y--")
    print(y)
    print("SHAPE: ", y.shape)
    # Testando modelos já implementados em bibliotecas SVM e MLP
    models = []
    models.append(('MLP', MLPClassifier()))
    models.append(('PolySVM', SVC(kernel="poly", gamma='scale')))
    models.append(('LinearSVM', SVC(kernel="linear", gamma='scale')))
    models.append(('RBF_SVM', SVC(kernel="rbf", gamma='scale')))
    # models.append(('ELM', ELM()))

    # Variáveis auxiliares
    results = []
    names = []
    print("--BENCHMARK--")
    # Laço de teste dos modelos
    for name, model in models:
        # Criando os Folds
        kfold = KFold(n_splits=5, random_state=42)
        # Recebendo os resultados obtidos pela validação cruzada
        cv_results = cross_val_score(model, X, y,
                                     cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        acc_mean = cv_results.mean()
        acc_std = cv_results.std()
        msg = "Acc({}) = {:.2f}±{:.2f}".format(name, acc_mean, acc_std)
        print(msg)

    plot(names, results)


if __name__ == '__main__':
    main()
