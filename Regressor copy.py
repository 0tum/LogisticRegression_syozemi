from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from dataset.mnist import load_mnist
import seaborn as sns
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, X_train, X_valid, y_train, y_valid):
        one = np.ones(X_train.shape[0]).reshape(X_train.shape[0], -1)
        self.X_train = np.column_stack((one, X_train))

        one = np.ones(X_valid.shape[0]).reshape(X_valid.shape[0], -1)
        self.X_valid = np.column_stack((one, X_valid))

        self.y_train = y_train
        self.y_valid = y_valid
        self.w = np.ones(28 * 28 + 1)

    def fit(self, eps, iteration):
        for i in range(iteration):
            difference = self.y_train - p(z(self.X_train, self.w))
            delta = -eps * (np.dot(-self.X_train.T, difference))
            self.w = self.w + delta

        self.p_train = p(z(self.X_train, self.w))
        self.p_valid = p(z(self.X_valid, self.w))

        return self.w

    def prediction(self):
        self.result_train = []
        self.result_valid = []

        for i in range(len(self.y_train)):
            if self.p_train[i] > 0.5:
                self.result_train.append(1)
            else:
                self.result_train.append(0)

        for i in range(len(self.y_valid)):
            if self.p_valid[i] > 0.5:
                self.result_valid.append(1)
            else:
                self.result_valid.append(0)

        # return self.result_train, self.result_valid #これだと予測したラベルリストが表示される。毎回表示させると煩雑なので非表示にしている

    #1つ目は学習用データに対する正解率、2つ目は検証用データに対する正解率
    def accuracy(self):
        count_train = 0
        count_valid = 0

        for i in range(len(self.y_train)):
            if self.result_train[i] == self.y_train[i]:
                count_train += 1

        for i in range(len(self.y_valid)):
            if self.result_valid[i] == self.y_valid[i]:
                count_valid += 1

        return count_train / len(self.y_train), count_valid / len(self.y_valid)

    def debug(self):
        # print(self.w)
        print(self.y_valid[0:10])
        print(self.result_valid[0:10])

        #検証用画像とそのラベル(Label:)、それに対する予測ラベル(赤字、下)を表示する
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes = axes.ravel()

        for i in range(10):
        # 画像を1次元から2次元に変形して表示
            image = self.X_valid[i][1:].reshape(28, 28)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f"Label: {self.y_valid[i]}")
            axes[i].axis('off')

            axes[i].text(13, 32, str(self.result_valid[i]), color='red')

        plt.tight_layout()
        plt.show()
        
        return "End"


def z(X, w):
    return np.dot(X, w)


def p(z_):
    return 1 / (1 + np.exp(-z_))


def load_iris_data():
    iris = load_iris()
    X = iris.data[0:100, :]
    y = iris.target[0:100]

    one = np.ones(100).reshape(100, -1)
    X = np.column_stack((one, X))

    return X, y


# mnistのデータをロードする。ゼロから作るDLで使われている関数を流用した。
(X_train, y_train), (X_valid, y_valid) = load_mnist(normalize=True, one_hot_label=True)

# 元のy_train, y_validはone-hotエンコーディングされたラベルなのでそれをスカラのラベルに戻る。
# 元のラベルの0番目の要素が1なら1、非1なら0とラベル付けする。
w_list=[]

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

    for j in range(60000):
        if y_train_0[j] == 1:
            X_train_norm.append(X_train[j])
            y_train_norm.append(1)
        elif j % 9 == 0:
            X_train_norm.append(X_train[j])
            if y_train_0[j] == 1:
                y_train_norm.append(1)
            else:
                y_train_norm.append(0)
        
    for j in range(10000):
        if y_valid_0[j] == 1:
            X_valid_norm.append(X_valid[j])
            y_valid_norm.append(1)
        elif j % 9 == 0:
            X_valid_norm.append(X_valid[j])
            if y_valid_0[j] == 1:
                y_valid_norm.append(1)
            else:
                y_valid_norm.append(0)

    X_train_norm = np.array(X_train_norm)
    X_valid_norm = np.array(X_valid_norm)

    lgr = LogisticRegression(X_train_norm, X_valid_norm, y_train_norm, y_valid_norm)
    w_list.append(lgr.fit(0.001, 10))

print(np.array(w_list).shape)
#w_list.shape = (10, 785)
#0行目が0orNot, 1行目が1orNot,...を判定するパラメータwになっている

