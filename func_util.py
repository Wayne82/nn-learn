import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
    :param z: Input value
    :return: Sigmoid of the input
    """
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    :param z: Input value
    :return: Derivative of the sigmoid
    """
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    """
    ReLU activation function.
    :param z: Input value
    :return: ReLU of the input
    """
    return np.maximum(0, z)

def relu_prime(z):
    """
    Derivative of the ReLU function.
    :param z: Input value
    :return: Derivative of the ReLU
    """
    return np.where(z > 0, 1, 0)

def quadratic_loss(y_true, y_pred):
    """
    Compute the quadratic loss.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Quadratic loss
    """
    return 0.5 * np.sum((y_pred - y_true) ** 2)

def quadratic_loss_prime(y_true, y_pred):
    """
    Compute the derivative of the quadratic loss.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Derivative of the quadratic loss
    """
    return y_pred - y_true