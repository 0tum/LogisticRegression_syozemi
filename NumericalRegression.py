from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from dataset.mnist import load_mnist
import seaborn as sns
import matplotlib.pyplot as plt

from Regressor import LogisticRegression

def z(X, w):
    return np.dot(X, w)

def p(z_):
    return 1 / (1 + np.exp(-z_))

def trainer(X_train, X_valid, y_train, y_valid):
    w = []

    #0~9について
    for i in range(10):
        y_train_0 = []
        y_valid_0 = []

        #数字についてone-hotエンコーディングからスカラの表現に戻す
        for j in range(60000):
            if y_train[j][i] != 1:
                y_train_0.append(0)
            else:
                y_train_0.append(1)
        for j in range(10000):
            if y_valid[j][i] != 1:
                y_valid_0.append(0)
            else:
                y_valid_0.append(1)

        X_train_norm=[]
        y_train_norm=[]
        X_valid_norm=[]
        y_valid_norm=[]

        for i in range(60000):
            if y_train_0[i] == 1:
                X_train_norm.append(X_train[i])
                y_train_norm.append(1)
            elif i % 9 == 0:
                X_train_norm.append(X_train[i])
                if y_train_0[i] == 1:
                    y_train_norm.append(1)
                else:
                    y_train_norm.append(0)
        
        for i in range(10000):
            if y_valid_0[i] == 1:
                X_valid_norm.append(X_valid[i])
                y_valid_norm.append(1)
            elif i % 9 == 0:
                X_valid_norm.append(X_valid[i])
                if y_valid_0[i] == 1:
                    y_valid_norm.append(1)
                else:
                    y_valid_norm.append(0)

        X_train_norm = np.array(X_train_norm)
        X_valid_norm = np.array(X_valid_norm)

        lgr = LogisticRegression(X_train_norm, X_valid_norm, y_train_norm, y_valid_norm)
        w.append(lgr.fit(0, 1000))

    return w
        



# mnistのデータをロードする。ゼロから作るDLで使われている関数を流用した。
(X_train, y_train), (X_valid, y_valid) = load_mnist(normalize=True, one_hot_label=True)

print(trainer(X_train, X_valid, y_train, y_valid))

#考え方
#1. 画像一枚を10個の分類器に入れる。(0orNot, 1orNot, ...)
#2. 10個の分類器からそれぞれ1つ確率が帰ってきて、合計10個の確率が得られる。
#3. 10個の確率のうち最も大きいものをその画像のラベルとする。

