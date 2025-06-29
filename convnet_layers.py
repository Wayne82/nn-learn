
import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)

    def forward(self, x):
        pass  # perform convolution

    def backward(self, grad):
        pass  # compute gradients and return grad to previous layer

    def update(self, lr, batch_size):
        pass  # gradient descent step