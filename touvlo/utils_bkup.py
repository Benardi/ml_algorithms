"""
.. module:: utils
    :synopsis: Provides routines of interest to different ML models.

.. moduleauthor:: Benardi Nunes <benardinunes@gmail.com>
"""

from numpy import zeros, copy, std, mean, float64, exp, seterr, maximum, ones_like

# sigmoid gradient function
def g(x, activation):
    """This function applies the given activation function on a given value.
    Default: Linear Activation f(x) = x

    :param x: Input value or object containing value.
    :type x: obj

    :param activation: Name of the activation function
    :type activation: str

    :returns: Activation function at value.
    :rtype: obj
    """
    a_lower = activation.lower()
    if a_lower == 'relu':
        return maximum(0, x)
    elif a_lower == 'tanh':
        return (exp(x) - exp(-1*x)) / (exp(x) + exp(-1*x))
    elif a_lower == 'sigmoid':
        return 1 / (1 + exp(-x))
    else:
        return x


# sigmoid gradient function
def g_grad(x, activation):
    """This function calculates the gradient of activation function at a given value.

    :param x: Input value or object containing value .
    :type x: obj

    :param activation: The activation function
    :type activation: str

    :returns: Activation function gradient at value.
    :rtype: obj
    """
    a_lower = activation.lower()
    if a_lower == 'relu':
        x[x <= 0] = 0
        x[x >= 0] = 1
        return x

    elif a_lower == 'tanh':
        activated_tanh = g(x, activation = 'tanh')
        return 1 - activated_tanh**2

    elif a_lower == 'sigmoid':
        return g(x, activation = 'sigmoid') * (1 - g(x, activation = 'sigmoid'))

    else:
        return ones_like(x)


def gradient_descent(X, y, grad, initial_theta,
                     alpha, num_iters, _lambda=None):
    """This function performs parameter optimization via gradient descent.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :param y: Column vector of expected values.
    :type y: numpy.array

    :param grad: Routine that generates the partial derivatives given theta.
    :type grad: numpy.array

    :param initial_theta: Initial value for parameters to be optimized.
    :type initial_theta: numpy.array

    :param alpha: Learning rate or step size of the optimization.
    :type alpha: float

    :param num_iters: Number of times the optimization will be performed.
    :type num_iters: int

    :param _lambda: Weight of the penalty term.
    :type _lambda: float

    :returns: Optimized model parameters.
    :rtype: numpy.array
    """
    if _lambda is not None:
        theta = copy(initial_theta)

        for _ in range(num_iters):
            theta = theta - alpha * grad(theta, X, y, _lambda)

    else:
        theta = copy(initial_theta)
        for _ in range(num_iters):
            theta = theta - alpha * grad(theta, X, y)

    return theta


def numerical_grad(J, theta, err):
    """Numerically calculates the gradient of a given cost function.

    :param J: Function handle that computes cost given theta.
    :type J: function

    :param theta: Model parameters.
    :type theta: numpy.array

    :param err: distance between points where J is evaluated.
    :type err: float

    :returns: Computed numeric gradient.
    :rtype: numpy.array
    """
    num_grad = zeros(theta.shape, dtype=float64)
    perturb = zeros(theta.shape, dtype=float64)

    for i in range(len(theta)):
        perturb[i] = err
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        num_grad[i] = (loss2 - loss1) / (2 * err)
        perturb[i] = 0

    return num_grad


def feature_normalize(X):
    """Performs Z score normalization in a numeric dataset.

    :param X: Features' dataset plus bias column.
    :type X: numpy.array

    :returns:
        - X_norm - Normalized features' dataset.
        - mu - Mean of each feature
        - sigma - Standard deviation of each feature.

    :rtype:
        - X_norm (:py:class: numpy.array)
        - mu (:py:class: numpy.array)
        - sigma (:py:class: numpy.array)
    """
    seterr(divide='ignore', invalid='ignore')
    mu = mean(X, axis=0)
    sigma = std(X, axis=0, ddof=1)

    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
