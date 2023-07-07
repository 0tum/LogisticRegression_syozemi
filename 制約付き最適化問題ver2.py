import numpy as np
import math
import random
from matplotlib import pyplot as plt
import pandas as pd

# minimize f = x1x2
# subject to x1 + x2 = 6


def f(x1, x2, alpha):
    z = x1 * x2 + (alpha / 2) * ((x1 + x2 - 6) ** 2)
    return z


def df(x1, x2, alpha):
    dzdx1 = x2 + alpha * (x1 + x2 - 6)
    dzdx2 = x1 + alpha * (x1 + x2 - 6)
    dz = np.array([dzdx1, dzdx2])
    return dz


def GradientDescent(x1_0, x2_0, alpha):
    eta = 0.1
    max_iteration = 1000
    # x1_pred = [x1_0]
    # x2_pred = [x2_0]

    for i in range(100):
        x1_0, x2_0 = np.array([x1_0, x2_0]) - eta * df(x1_0, x2_0, alpha)

    des = df(x1_0, x2_0, alpha)
    norm = math.sqrt(df[0] ** 2 + df[1] ** 2)

    if norm < eps_bar:
        x1_star = x1_0
        x2_star = x2_0
        return x1_star, x2_star

    else:
        alpha *= 10


# さすがに手におえない。これはやらない


alpha = 100
eps_bar = 0.1
# x1_0 = random.uniform(-100, 100)
# x2_0 = random.uniform(-100, 100)

rates = np.ones(10) * 2
rates = rates ** np.arange(10)

alpha_num = []
x1_pred = []
x2_pred = []

for num_alpha in rates:
    # x1_0 = random.uniform(-1, 1)
    # x2_0 = random.uniform(-1, 1)

    x1_0 = 1
    x2_0 = 1

    a, b = GradientDescent(x1_0, x2_0, num_alpha)
    alpha_num.append(num_alpha)
    x1_pred.append(a)
    x2_pred.append(b)
