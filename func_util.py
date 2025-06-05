import numpy as np

class ActivationFunction:
    """
    Enum for activation functions.
    """
    SIGMOID = 1
    RELU = 2
    SOFTMAX = 3

    @staticmethod
    def get_activation_function(activation):
        """
        Get the activation function class based on the provided type.
        :param activation: Type of activation function
        :return: Corresponding activation function class
        """
        if activation == ActivationFunction.SIGMOID:
            return Sigmoid
        elif activation == ActivationFunction.RELU:
            return ReLU
        elif activation == ActivationFunction.SOFTMAX:
            return Softmax
        else:
            raise ValueError("Invalid activation function type.")

class CostFunction:
    """
    Enum for cost functions.
    """
    QUADRATIC = 1
    CROSS_ENTROPY = 2

    @staticmethod
    def get_cost_function(cost):
        """
        Get the cost function class based on the provided type.
        :param cost: Type of cost function
        :return: Corresponding cost function class
        """
        if cost == CostFunction.QUADRATIC:
            return QuadraticLoss
        elif cost == CostFunction.CROSS_ENTROPY:
            return CrossEntropyLoss
        else:
            raise ValueError("Invalid cost function type.")

class Sigmoid:
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z):
        s = Sigmoid.fn(z)
        return s * (1 - s)

    @staticmethod
    def initialize_weights_coefficient(x):
        return np.sqrt(1.0 / x)

class ReLU:
    @staticmethod
    def fn(z):
        return np.maximum(0, z)

    @staticmethod
    def prime(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def initialize_weights_coefficient(x):
        return np.sqrt(2.0 / x)  # He initialization for ReLU

class Softmax:
    @staticmethod
    def fn(z):
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        return exp_z / np.sum(exp_z)

    @staticmethod
    def prime(z):
        s = Softmax.fn(z)
        return np.diag(s) - np.outer(s, s)

    @staticmethod
    def initialize_weights_coefficient(x):
        return np.sqrt(1.0 / x)

class QuadraticLoss:
    @staticmethod
    def type():
        return CostFunction.QUADRATIC

    @staticmethod
    def fn(y_true, y_pred):
        return 0.5 * np.sum((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

class CrossEntropyLoss:
    @staticmethod
    def type():
        return CostFunction.CROSS_ENTROPY

    @staticmethod
    def fn(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-15))  # add small value to avoid log(0)

    @staticmethod
    def prime(y_true, y_pred):
        return -y_true / (y_pred + 1e-15)  # add small value to avoid division by zero

    @staticmethod
    def delta(y_true, y_pred):
        """
        Compute the gradient of the cross-entropy loss with respect to activations of the softmax layer.
        :param y_true: True labels (one-hot encoded)
        :param y_pred: Predicted probabilities from softmax
        :return: Gradient of the loss with respect to activations of the softmax layer
        """
        return y_pred - y_true
