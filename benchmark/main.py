# coding: utf-8

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

import dataset
from elm import ELM
from nnrbf import NNRBF

plt.style.use('grayscale')


def plot(names, results, ylabel="Acurácia"):
    # Construção gráfica da box, contendo o comparativo entre os aloritimos
    fig = plt.figure()
    fig.suptitle('Comparação dos algoritimos')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.xlabel("Modelos")
    plt.ylabel(ylabel)
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
    q = 200
    models.append(('MLP', MLPClassifier((q,))))
    models.append(('ELM', ELM(q=q)))
    # models.append(('NN_RBF', NNRBF(q=500)))
    models.append(('LinearSVM', SVC(kernel="linear", gamma='scale')))
    models.append(('PolySVM', SVC(kernel="poly", gamma='scale')))
    models.append(('RBF_SVM', SVC(kernel="rbf", gamma='scale')))

    k = 5
    # Variáveis auxiliares
    accs = []
    names = []
    print("--BENCHMARK--")
    # Laço de teste dos modelos
    for name, model in models:
        # Criando os Folds
        kfold = KFold(n_splits=k,
                      random_state=42)
        # Recebendo os resultados obtidos pela validação cruzada
        cv_accs = cross_val_score(model, X, y,
                                  cv=kfold, scoring='accuracy')
        accs.append(cv_accs)
        names.append(name)
        acc_mean = cv_accs.mean()
        acc_std = cv_accs.std()
        msg = "Acc({}) = {:.2f}±{:.2f}".format(name, acc_mean, acc_std)
        print(msg)

    plot(names, accs)


if __name__ == '__main__':
    main()
