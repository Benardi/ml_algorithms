"""
.. module:: nn_clsf_vect
    :synopsis: Provides routines to construct a Classification Neural Network.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy.random import seed, randn
from numpy import (dot, log, divide, zeros, squeeze)
from numpy import sum as add

from touvlo.utils import sigmoid, sigmoid_backward, relu, relu_backward


def init_params(layer_dims, _seed=1):

    seed(_seed)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = randn(layer_dims[l],
                                         layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = zeros(shape=(layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    # number of layers in the neural network
    L = len(parameters) // 2

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A

        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)],
                                          parameters['b' + str(L)], "sigmoid")

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = -(1 / m) * add(Y * log(AL) + (1 - Y) * log(1 - AL))

    # this turns [[17]] into 17
    cost = squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * dot(dZ, A_prev.T)
    db = (1 / m) * add(dZ, axis=1, keepdims=True)
    dA_prev = dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (divide(Y, AL) - divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients.
    # Inputs: "dAL, current_cache".
    # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    (grads["dA" + str(L - 1)],
     grads["dW" + str(L)],
     grads["db" + str(L)]) = linear_activation_backward(dAL,
                                                        current_cache,
                                                        "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache".
        # Outputs: "grads["dA" + str(l)] ,
        #           grads["dW" + str(l + 1)],
        #           grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * \
            grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * \
            grads["db" + str(l + 1)]

    return parameters
