import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
def main():
    # Abrindo dataset MNIST
    # REF: "https://datahub.io/machine-learning/optdigits#readme"
    df = pd.read_csv("mnist.csv")
    # Separando atributos X e Y
    X = df.drop("class",axis = 1)
    Y = df["class"]

    # Aplicando Hold-out 70 - 30
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.5)

    # Testando ELM
    
    # Testando modelos já implementados em bibliotecas SVM e MLP
    models = []
    models.append(('MLP', MLPClassifier()))
    models.append(('SVM', SVC(kernel = "poly")))
    
    # Variáveis auxiliares
    results = []
    names = []

    # Laço de teste dos modelos
    for name, model in models:
        # Criando os Folds
        kfold = KFold(n_splits=5, random_state=42)
        # Recebendo os resultados obtidos pela validação cruzada
        cv_results = cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "{}: {} ({})".format(name, cv_results.mean(), cv_results.std())
        print(msg)
    # Construção gráfica da box, contendo o comparativo entre os aloritimos
    fig = plt.figure()
    fig.suptitle('Comparação dos algoritimos')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.xlabel("Modelos")
    plt.ylabel("Acurácia")
    plt.show()
