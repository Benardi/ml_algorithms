from math import sqrt

from numpy.random import uniform
from numpy import (float64, ones, append, sum, exp, dot,
                   log, power, zeros, reshape, empty)


# sigmoid gradient function
def g(x):
    return 1 / (1 + exp(-x))


# sigmoid gradient function
def g_grad(x):
    return g(x) * (1 - g(x))


def cost_function(X, y, theta, _lambda, num_labels, n_hidden_layers=1):
    m, n = X.shape
    intercept = ones((m, 1), dtype=float64)
    X = append(intercept, X, axis=1)

    z, a = feed_forward(X, theta, n_hidden_layers)
    L = n_hidden_layers + 1  # last layer

    J = 0
    for c in range(num_labels):
        _J = dot(1 - (y == c).T, log(1 - a[L][:, c]))
        _J = _J + dot((y == c).T, log(a[L][:, c]))
        J = J - (1 / m) * sum(_J)

    theta_squared_term = 0
    for j in range(len(theta)):
        theta_squared_term += sum(power(theta[j][:, 1:], 2))

    J = J + (_lambda / (2 * m)) * theta_squared_term

    return J


def unravel_params(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, n_hidden_layers=1):

    input_layer_n_units = hidden_layer_size * (input_layer_size + 1)
    hidden_layer_n_units = hidden_layer_size * (hidden_layer_size + 1)

    theta = empty((n_hidden_layers + 1), dtype=object)

    # input layer to hidden layer
    theta[0] = nn_params[0:input_layer_n_units]
    theta[0] = reshape(theta[0], (hidden_layer_size, (input_layer_size + 1)))

    # hidden layer to hidden layer
    for i in range(1, n_hidden_layers):

        start = input_layer_n_units + (i - 1) * hidden_layer_n_units
        end = input_layer_n_units + i * hidden_layer_n_units

        theta[i] = nn_params[start:end]
        theta[i] = reshape(
            theta[i], (hidden_layer_size, (hidden_layer_size + 1)))

    # hidden layer to output layer
    start = input_layer_n_units + (n_hidden_layers - 1) * hidden_layer_n_units

    theta[n_hidden_layers] = nn_params[start:]
    theta[n_hidden_layers] = reshape(theta[n_hidden_layers],
                                     (num_labels, (hidden_layer_size + 1)))

    return theta


def feed_forward(X, theta, n_hidden_layers=1):
    z = empty((n_hidden_layers + 2), dtype=object)
    a = empty((n_hidden_layers + 2), dtype=object)

    # Input layer
    a[0] = X

    # Hidden unit layers
    for l in range(1, (len(a) - 1)):
        z[l] = a[l - 1].dot(theta[l - 1].T)
        a[l] = g(z[l])
        a[l] = append(ones((len(a[l]), 1), float64),  # add intercept
                      a[l], axis=1)

    # Output layer
    z[len(a) - 1] = a[(len(a) - 2)].dot(theta[(len(a) - 2)].T)
    a[len(a) - 1] = g(z[len(a) - 1])  # hypothesis

    return z, a


def back_propagation(theta, a, z, num_labels, y, n_hidden_layers=1):
    delta = empty((n_hidden_layers + 2), dtype=object)
    L = n_hidden_layers + 1  # last layer
    delta[L] = zeros(shape=a[L].shape, dtype=float64)

    for c in range(num_labels):
        delta[L][:, c] = a[L][:, c] - (y == c)

    for l in range(L, 1, -1):
        delta[l - 1] = delta[l].dot(theta[l - 1])[:, 1:] * g_grad(z[l - 1])

    return delta


def grad(nn_params, X, y, _lambda, input_layer_size,
         num_labels, hidden_layer_size, n_hidden_layers=1):

    theta = unravel_params(nn_params, input_layer_size, hidden_layer_size,
                           num_labels, n_hidden_layers)

    # Initi gradient with zeros
    theta_grad = empty((n_hidden_layers + 1), dtype=object)
    for i in range(len(theta)):
        theta_grad[i] = zeros(shape=theta[i].shape, dtype=float64)

    m, n = X.shape
    intercept = ones((m, 1), dtype=float64)
    X = append(intercept, X, axis=1)

    for t in range(m):

        z, a = feed_forward(X[[t], :], theta, n_hidden_layers)
        delta = back_propagation(theta, a, z, num_labels,
                                 y[t, :], n_hidden_layers)

        for l in range(len(theta_grad)):
            theta_grad[l] = theta_grad[l] + dot(delta[l + 1].T, a[l])

    for i in range(len(theta_grad)):
        theta_grad[i] = (1 / m) * theta_grad[i]

    # regularization
    for i in range(len(theta_grad)):
        theta_grad[i][:, 1:] = theta_grad[i][:, 1:] + \
            (_lambda / m) * theta[i][:, 1:]

    flat_theta_grad = append(theta_grad[0].flatten(), theta_grad[1].flatten())
    for i in range(2, len(theta_grad)):
        flat_theta_grad = append(flat_theta_grad, theta_grad[i].flatten())

    return flat_theta_grad


def rand_init_weights(L_in, L_out):
    W = zeros((L_out, 1 + L_in), float64)  # plus 1 for bias term
    epsilon_init = sqrt(6) / sqrt((L_in + 1) + L_out)

    W = uniform(size=(L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return W


def init_nn_weights(n_hidden_layers, input_layer_size,
                    hidden_layer_size, num_labels):

    theta = empty((n_hidden_layers + 1), dtype=object)
    theta[0] = rand_init_weights(input_layer_size, hidden_layer_size)

    for l in range(1, n_hidden_layers):
        theta[l] = rand_init_weights(hidden_layer_size, hidden_layer_size)

    theta[n_hidden_layers] = rand_init_weights(hidden_layer_size, num_labels)

    return theta
