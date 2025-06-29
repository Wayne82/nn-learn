
import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights and biases
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels,))

        # Buffer for gradients
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        x shape: (C_in, H_in, W_in)
        output shape: (C_out, H_out, W_out)
        """
        self.x = x
        _, H_in, W_in = x.shape
        C_out, _, kernel_size, _ = self.W.shape
        H_out = H_in - kernel_size + 1
        W_out = W_in - kernel_size + 1
        out = np.zeros((C_out, H_out, W_out))

        for c in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    local_region = x[:, h:h + kernel_size, w:w + kernel_size]
                    out[c, h, w] = np.sum(local_region * self.W[c]) + self.b[c]

        return out

    def backward(self, grad):
        pass  # compute gradients and return grad to previous layer

    def update(self, lr, batch_size):
        pass  # gradient descent step