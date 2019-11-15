from numpy import log, zeros, dot
from numpy import sum as add

from touvlo.utils import g


def h(X, w, b):
    return g(dot(w.T, X) + b)


def cost_func(X, Y, hyp=None, **kwargs):
    if hyp is None:
        hyp = h(X, **kwargs)

    _, m = Y.shape
    J = -(1 / m) * add((Y * log(hyp) + (1 - Y) * log(1 - hyp)))

    return J


def grad(X, Y, w, b):

    _, m = X.shape
    A = h(X, w, b)  # compute activation

    dz = A - Y
    dw = (1 / m) * dot(X, dz.T)
    db = (1 / m) * add(dz)

    return dw, db


def predict(X, w, b):
    n, m = X.shape
    Y_hat = zeros((1, m))
    w = w.reshape(n, 1)

    A = h(X, w, b)

    for i in range(A.shape[1]):
        if A[:, i] >= 0.5:
            Y_hat[:, i] = 1
        else:
            Y_hat[:, i] = 0

    return Y_hat
