from sklearn.datasets import load_iris
import numpy as np

def z(X, w):
    return np.dot(X, w)

def p(z):
    return 1/(1+np.exp(-z))

def estimate(X, w, y):
    eps = 0.001
    result=[]

    for i in range(100):
        res = y - p(z(X, w))
        delta = -eps*(np.dot(-X.T, res))
        w = w + delta
    
    return p(z(X, w))

def prediction(p):
    result = []

    for i in range(100):
        if p[i] >= 0.5:
            result.append(1)
        else:
            result.append(0)
    
    return result

def accuracy(res, y):
    cor = 0
    for i in range(100):
        if res[i] == y[i]:
            cor+=1
    
    return float(cor/100)

iris = load_iris()
data = iris.data
X = data[0:100, 0:2]
y = iris.target[0:100]

one = np.ones(100).reshape(100, -1)

X = np.column_stack((one, X))

w = [1, 1, 1]

p = estimate(X, w, y)
res = prediction(p)

print(p)
print(res)
print(accuracy(res, y))
