from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

def z(X, w):
    return np.dot(X, w)

def p(zz):
    return 1/(1+np.exp(-zz))

def estimate(X, y, w):
    eps = 0.001

    for i in range(10000):
        res = y - p(z(X, w))
        delta = -eps*(np.dot(-X.T, res))
        w = w + delta
    
    return p(z(X, w)), w

def convert(p):
    result = []

    for i in range(len(p)):
        if p[i] >= 0.5:
            result.append(1)
        else:
            result.append(0)
    
    return result

def accuracy(res, y):
    cor = 0
    for i in range(len(y)):
        if res[i] == y[i]:
            cor+=1
    
    return float(cor/len(y))

def prediction(X, y, w):
    proba = p(z(X, w))

    res = convert(proba)
    acc = accuracy(res, y) 

    return acc
 
def load_iris_data():
    iris = load_iris()
    X = iris.data[0:100, :]
    y = iris.target[0:100]

    one =  np.ones(100).reshape(100, -1)
    X = np.column_stack((one, X))

    return X, y

X, y = load_iris_data()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

w = [1, 1, 1, 1, 1]

p, w = estimate(X_train, y_train, w)
y_pred = convert(p)

print("Train")
print(accuracy(y_pred, y_train))
print(w) 

print("Validation")
prob = 1/(1+np.exp(-z(X_valid, w))) #謎のエラー
print(accuracy(convert(prob), y_valid))

print(y_valid)
print(convert(prob))