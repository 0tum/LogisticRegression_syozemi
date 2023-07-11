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

    def fit(self, target, eps, iteration, epoch):
        self.y_train_0 = []
        self.y_valid_0 = []

        # 数字についてone-hotエンコーディングからスカラの表現に戻す
        for j in range(60000):
            if self.y_train[j][target] != 1:
                self.y_train_0.append(0)
            else:
                self.y_train_0.append(1)
        for j in range(10000):
            if self.y_valid[j][target] != 1:
                self.y_valid_0.append(0)
            else:
                self.y_valid_0.append(1)

        self.X_train_norm = []
        self.y_train_norm = []
        self.X_valid_norm = []
        self.y_valid_norm = []

        for j in range(60000):
            if self.y_train_0[j] == 1:
                self.X_train_norm.append(self.X_train[j])
                self.y_train_norm.append(1)
            elif j % 9 == 0:
                self.X_train_norm.append(self.X_train[j])
                if self.y_train_0[j] == 1:
                    self.y_train_norm.append(1)
                else:
                    self.y_train_norm.append(0)

        for j in range(10000):
            if self.y_valid_0[j] == 1:
                self.X_valid_norm.append(X_valid[j])
                self.y_valid_norm.append(1)
            elif j % 9 == 0:
                self.X_valid_norm.append(X_valid[j])
                if self.y_valid_0[j] == 1:
                    self.y_valid_norm.append(1)
                else:
                    self.y_valid_norm.append(0)

        self.X_train_norm = np.array(self.X_train_norm)
        self.X_valid_norm = np.array(self.X_valid_norm)

        for i in range(iteration):
            difference = self.y_train_norm - p(z(self.X_train_norm, self.w))
            delta = -eps * (np.dot(-self.X_train_norm.T, difference))
            self.w = self.w + delta

            if np.linalg.norm(difference) <= 10:
                print(i)
                break

        print(epoch + 1, " /10 Finished")
        return self.w

    def debug(self):
        pass


class LGRCalcucation:
    def __init__(self, X_train, X_valid, y_train, y_valid, w_list):
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.w_list = w_list

        self.y_train_10 = []
        self.y_valid_10 = []

        for i in range(self.y_train.shape[0]):
            self.y_train_10.append(np.argmax(self.y_train[i]))
        for i in range(self.y_valid.shape[0]):
            self.y_valid_10.append(np.argmax(self.y_valid[i]))

        one = np.ones(self.X_train.shape[0]).reshape(self.X_train.shape[0], -1)
        self.X_train = np.column_stack((one, self.X_train))

        one = np.ones(self.X_valid.shape[0]).reshape(self.X_valid.shape[0], -1)
        self.X_valid = np.column_stack((one, self.X_valid))

    def predict(self):
        self.result_train = []
        self.result_valid = []

        self.p_train = p(z(self.X_train, np.array(self.w_list).T))
        self.p_valid = p(z(self.X_valid, np.array(self.w_list).T))

        for i in range(self.p_train.shape[0]):
            self.result_train.append(np.argmax(self.p_train[i]))
        for i in range(self.p_valid.shape[0]):
            self.result_valid.append((np.argmax(self.p_valid[i])))

        return self.p_train, self.p_valid, self.result_train, self.result_valid

    def accuracy(self):
        count_train = 0
        for i in range(len(self.y_train_10)):
            if self.y_train_10[i] == self.result_train[i]:
                count_train += 1

        count_valid = 0
        for i in range(len(self.y_valid)):
            if self.y_valid_10[i] == self.result_valid[i]:
                count_valid += 1

        return count_train / len(self.y_train_10), count_valid / len(
            self.y_valid_10
        )

    def debug(self):
        return self.y_valid_10


def z(X, w):
    return np.dot(X, w)


def p(z_):
    return 1 / (1 + np.exp(-z_))


# mnistのデータをロードする。ゼロから作るDLで使われている関数を流用した。
(X_train, y_train), (X_valid, y_valid) = load_mnist(
    normalize=True, one_hot_label=True
)

# 元のy_train, y_validはone-hotエンコーディングされたラベルなのでそれをスカラのラベルに戻る。
# 元のラベルの0番目の要素が1なら1、非1なら0とラベル付けする。
w_list = []


for i in range(10):
    lgr = LogisticRegression(X_train, X_valid, y_train, y_valid)
    w_list.append(lgr.fit(i, 0.001, 175, epoch=i))

# w_list.shape = (10, 785)
# 0行目が0orNot, 1行目が1orNot,...を判定するパラメータwになっている


lgrC = LGRCalcucation(X_train, X_valid, y_train, y_valid, w_list)
p_train, p_valid, result_train, result_valid = lgrC.predict()
y_valid_10 = lgrC.debug()
print(lgrC.accuracy())

pickup = np.random.randint(0, 10001, size=10)

print("answer    :", y_valid_10[0:10])
print("prediction:", result_valid[0:10])


fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.ravel()


for i in range(10):
    j = pickup[i]
    # 画像を1次元から2次元に変形して表示
    image = X_valid[j].reshape(28, 28)
    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(f"prediction: {result_valid[j]}")
    axes[i].text(
        7, image[i].shape[0] + 5, f"answer:{y_valid_10[j]}", color="black"
    )
    axes[i].axis("off")

plt.tight_layout()
plt.show()

