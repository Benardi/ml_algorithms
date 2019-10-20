from numpy import power, multiply, where, zeros
from numpy import sum as add


def cost_function(X, R, Y, theta, _lambda):
    J = power(X.dot(theta.T) - Y, 2)
    J = (1 / 2) * add(multiply(J, R))
    J = J + (_lambda / 2) * add(power(theta, 2))
    J = J + (_lambda / 2) * add(power(X, 2))
    return J


def grad(X, R, Y, theta, num_users, num_movies, num_features, _lambda):

    X_grad = zeros(X.shape)
    theta_grad = zeros(theta.shape)

    for i in range(num_movies):
        idx = where(R[i, :] == 1)  # users that have rated product i
        X_grad[i, :] = (X[i, :].dot(theta[idx].T)
                        - Y[i, idx[0]]).dot(theta[idx])
        X_grad[i, :] = X_grad[i, :] + _lambda * X[i, :]

    for j in range(num_users):
        idx = where(R[:, j] == 1)  # products that have been rated by user j
        theta_grad[j, :] = (theta[j, :].dot(
            X[idx].T) - Y[idx[0], j]).dot(X[idx])
        theta_grad[j, :] = theta_grad[j, :] + _lambda * theta[j, :]

    return X_grad, theta_grad
