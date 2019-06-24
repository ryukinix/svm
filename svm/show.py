import numpy as np
from matplotlib import pyplot as plt
import dataset

plt.style.use('grayscale')


def plot_digits(X, y, shape=(8, 8)):
    n = len(X)
    idx = np.arange(0, n)
    sampling = np.random.choice(idx, 8)
    X_sample = X[sampling]
    y_sample = y[sampling]
    subplot = 241
    for x, label in zip(X_sample, y_sample):
        ax = plt.subplot(subplot)
        title = "class: {}".format(label)
        image = x.reshape(shape)
        ax.set_title(title)
        plt.imshow(image)
        subplot += 1
    plt.show()


def main():
    X, y = dataset.digits()
    plot_digits(X, y)


if __name__ == '__main__':
    main()
