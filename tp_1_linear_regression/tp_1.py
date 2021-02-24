"""
Name : tp_1.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 22/02/21 02:35 م
Desc:
"""
from PIL import Image
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def pretraitement(path):
    img = Image.open(path)
    img = np.array(img)

    # reperer le pixel le plus bas:
    result = np.where(img == 255)
    rows = result[0]
    columns = result[1]
    y = np.max(rows)
    x = columns[np.argmax(rows)]
    return x, y


def get_paths(path):
    imgs = []
    import os
    for file in os.listdir(path):
        if file.endswith(".png"):
            imgs.append(os.path.join(path, file))
    return imgs


def get_data(path):
    imgs = get_paths(path)
    yy = []
    xx = []

    for img in imgs:
        x, y = pretraitement(img)
        xx.append(x)
        yy.append(y)
    return np.array(xx), np.array(yy)


if __name__ == '__main__':
    # pipline:
    # 1) définir: régression

    # 2) préparation des données:
    x, y = get_data("dataset/train/")
    x = x.reshape(-1, 1)

    # 3) select model:
    # prepare regression:
    model = LinearRegression()
    model.fit(x, y)
    print('Coefficients: ', model.coef_)

    # 4) tester le modele et evaluer sur l'ensemble de test:

    # 5) evaluer:
    y_pred = model.predict(x)
    print('Mean squared error: %.2f' % mean_squared_error(y, y_pred))
    print('Coefficient of determination: %.2f\n' % r2_score(y, y_pred))

    # 6) visualization: train-set
    plt.scatter(x, y, color='black')
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regression (train)')
    plt.show()

    # visualization: test-set

    # essayer de comparer les résultats:
