
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
        """
        grad shape: (C_out, H_out, W_out)
        dx shape: (C_in, H_in, W_in)
        """
        _, H_in, W_in = self.x.shape
        C_out, H_out, W_out = grad.shape
        _, _, K, _ = self.W.shape

        dx = np.zeros_like(self.x)

        for c in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    local_region = self.x[:, h:h + K, w:w + K]
                    self.dW[c] += local_region * grad[c, h, w]
                    self.db[c] += grad[c, h, w]
                    dx[:, h:h + K, w:w + K] += self.W[c] * grad[c, h, w]

        return dx

    def update(self, lr, batch_size):
        self.W -= lr * self.dW / batch_size
        self.b -= lr * self.db / batch_size

    def zero_grad(self):
        """
        Reset gradients to zero.
        This is useful before starting a new batch.
        """
        self.dW.fill(0)
        self.db.fill(0)

class ReLu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        dx = grad * (self.x > 0)
        return dx

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.indices = None

    def forward(self, x):
        self.input = x
        C, H_in, W_in = x.shape
        H_out = (H_in - self.pool_size) // self.stride + 1
        W_out = (W_in - self.pool_size) // self.stride + 1

        out = np.zeros((C, H_out, W_out))
        self.indices = np.zeros_like(x, dtype=bool)

        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    local_region = x[c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                    max_val = np.max(local_region)
                    out[c, h, w] = max_val
                    self.indices[c, h_start:h_start + self.pool_size,
                                 w_start:w_start + self.pool_size] |= (local_region == max_val)

        return out

    def backward(self, grad):
        C, H_in, W_in = self.input.shape
        dx = np.zeros_like(self.input)
        out_h, out_w = grad.shape

        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * self.stride
                    w_start = w * self.stride
                    mask = self.indices[c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                    dx[c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size] += grad[c, h, w] * mask

        return dx

class Flatten:
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(-1, 1)

    def backward(self, grad):
        return grad.reshape(self.x_shape)

class FullyConnected:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and biases
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros((out_features,))

        # Buffer for gradients
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return np.dot(self.W, x) + self.b

    def backward(self, grad):
        self.dW += np.dot(grad, self.x.T)
        self.db += grad
        dx = np.dot(self.W.T, grad)
        return dx

    def update(self, lr, batch_size):
        self.W -= lr * self.dW / batch_size
        self.b -= lr * self.db / batch_size

    def zero_grad(self):
        """
        Reset gradients to zero.
        This is useful before starting a new batch.
        """
        self.dW.fill(0)
        self.db.fill(0)

class CrossEntropyLoss:
    def __init__(self):
        '''
        Initialize the CrossEntropyLoss layer.
        This layer computes the cross-entropy loss and its gradients.
        '''
        self.logits = None
        self.labels = None
        self.probs = None

    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels

        # Compute softmax probabilities
        exps = np.exp(logits - np.max(logits))
        self.probs = exps / np.sum(exps)

        # Add small value to avoid log(0)
        probs = -np.log(self.probs + 1e-10)

        # Compute cross-entropy loss
        loss = np.sum(labels * probs)
        return loss

    def backward(self):
        '''
        Compute the gradient of the loss with respect to the logits.
        :return: Gradient of the loss with respect to the logits.
        '''
        # ∂L/∂z = probs - target
        # (same as gradient from softmax + cross-entropy combo)
        grad = self.probs - self.labels
        return grad