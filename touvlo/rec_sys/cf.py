from numpy import power, multiply, where, zeros, reshape, append
from numpy import sum as add


def unravel_params(params, num_users, num_products, num_features):
    """Unravels flattened array into features' matrices

    :param params: Row vector of coefficients.
    :type params: numpy.array

    :param num_users: Number of users in this instance.
    :type num_users: int

    :param num_products: Number of products in this instance.
    :type num_products: int

    :param num_features: Number of features in this instance.
    :type num_features: int

    :returns:
        - X - Matrix of product features.
        - theta - Matrix of user features.

    :rtype:
        - X (:py:class: numpy.array)
        - theta (:py:class: numpy.array)
    """
    X = params[0:(num_products * num_features)]
    X = reshape(X, (num_products, num_features))
    theta = params[(num_products * num_features):]
    theta = reshape(theta, (num_users, num_features))
    return X, theta


def cost_function(X, R, Y, theta, _lambda):
    """Computes the cost function J for Collaborative Filtering.

    :param X: Matrix of product features.
    :type X: numpy.array

    :param Y: Scores' dataset.
    :type Y: numpy.array

    :param R: Dataset of 0s and 1s (whether there's a rating).
    :type R: numpy.array

    :param theta: Matrix of user features.
    :type theta: numpy.array

    :param _lambda: The regularization hyperparameter.
    :type _lambda: float

    :returns: Computed cost.
    :rtype: float
    """
    J = power(X.dot(theta.T) - Y, 2)
    J = (1 / 2) * add(multiply(J, R))
    J = J + (_lambda / 2) * add(power(theta, 2))
    J = J + (_lambda / 2) * add(power(X, 2))
    return J


def grad(params, R, Y, num_users, num_products, num_features, _lambda):
    """Computes the gradient for Collaborative Filtering.

    :param params: flattened product and user features.
    :type params: numpy.array

    :param Y: Scores' dataset.
    :type Y: numpy.array

    :param R: Dataset of 0s and 1s (whether there's a rating).
    :type R: numpy.array

    :param num_users: Number of users in this instance.
    :type num_users: int

    :param num_products: Number of products in this instance.
    :type num_products: int

    :param num_features: Number of features in this instance.
    :type num_features: int

    :param _lambda: The regularization hyperparameter.
    :type _lambda: float

    :returns: Flattened gradient for product and user features.
    :rtype: numpy.array
    """
    X, theta = unravel_params(params, num_users, num_products, num_features)
    X_grad = zeros(X.shape)
    theta_grad = zeros(theta.shape)

    for i in range(num_products):
        idx = where(R[i, :] == 1)  # users that have rated product i
        X_grad[i, :] = (X[i, :].dot(theta[idx].T)
                        - Y[i, idx[0]]).dot(theta[idx])
        X_grad[i, :] = X_grad[i, :] + _lambda * X[i, :]

    for j in range(num_users):
        idx = where(R[:, j] == 1)  # products that have been rated by user j
        theta_grad[j, :] = (theta[j, :].dot(
            X[idx].T) - Y[idx[0], j]).dot(X[idx])
        theta_grad[j, :] = theta_grad[j, :] + _lambda * theta[j, :]

    flat_params = append(X_grad.flatten(), theta_grad.flatten())
    return flat_params
