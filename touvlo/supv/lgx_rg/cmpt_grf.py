from numpy import log, zeros, dot, squeeze
from numpy import sum as add

from touvlo.utils import g


def h(X, w, b):
    return g(dot(w.T, X) + b)


def cost_func(X, Y, h):
    m = len(Y)
    J = -(1 / m) * add((Y * log(h) + (1 - Y) * log(1 - h)))

    return J


def grad(w, b, X, Y):

    m, _ = X.shape
    A = h(X, w, b)  # compute activation
    cost = cost_func(X, Y, A)

    dz = A - Y
    dw = (1 / m) * dot(X, dz.T)
    db = (1 / m) * add(dz)
    cost = squeeze(cost)

    return dw, db


def predict(w, b, X):
    m, _ = X.shape
    Y_prediction = zeros((1, m))
    w = w.reshape(m, 1)

    A = h(X, w, b)

    for i in range(A.shape[1]):
        if A[:, i] >= 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    return Y_prediction
