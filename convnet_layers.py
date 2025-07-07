import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights and biases
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.random.randn(out_channels, 1)

        # Buffer for gradients
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        x shape: (B, C_in, H_in, W_in)
        output shape: (B, C_out, H_out, W_out)
        """
        self.x = x
        B, _, H_in, W_in = x.shape
        C_out, _, kernel_size, _ = self.W.shape
        H_out = H_in - kernel_size + 1
        W_out = W_in - kernel_size + 1
        out = np.zeros((B, C_out, H_out, W_out))

        for b in range(B):
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        local_region = x[b, :, h:h + kernel_size, w:w + kernel_size]
                        out[b, c, h, w] = np.sum(local_region * self.W[c]) + self.b[c, 0]

        return out

    def backward(self, grad):
        """
        grad shape: (B, C_out, H_out, W_out)
        dx shape: (B, C_in, H_in, W_in)
        """
        B, _, H_in, W_in = self.x.shape
        _, _, H_out, W_out = grad.shape
        C_out, _, K, _ = self.W.shape

        dx = np.zeros_like(self.x)

        for b in range(B):
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        local_region = self.x[b, :, h:h + K, w:w + K]
                        self.dW[c] += local_region * grad[b, c, h, w]
                        self.db[c, 0] += grad[b, c, h, w]
                        dx[b, :, h:h + K, w:w + K] += self.W[c] * grad[b, c, h, w]

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
        """
        Forward pass for max pooling layer.
        :param x: Input tensor of shape (B, C, H_in, W_in)
        :return: Output tensor of shape (B, C, H_out, W_out)
        """
        self.input = x
        B, C, H_in, W_in = x.shape
        H_out = (H_in - self.pool_size) // self.stride + 1
        W_out = (W_in - self.pool_size) // self.stride + 1

        out = np.zeros((B, C, H_out, W_out))
        self.indices = np.zeros_like(x, dtype=bool)

        for b in range(B):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        local_region = x[b, c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                        max_val = np.max(local_region)
                        out[b, c, h, w] = max_val
                        self.indices[b, c, h_start:h_start + self.pool_size,
                                 w_start:w_start + self.pool_size] |= (local_region == max_val)

        return out

    def backward(self, grad):
        """
        Backward pass for max pooling layer.
        :param grad: Gradient tensor of shape (B, C, H_out, W_out)
        :return: Gradient tensor of shape (B, C, H_in, W_in)
        """
        B, C, H_in, W_in = self.input.shape
        _, _, out_h, out_w = grad.shape

        dx = np.zeros_like(self.input)

        for b in range(B):
            for c in range(C):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        mask = self.indices[b, c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                        dx[b, c, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size] += grad[b, c, h, w] * mask

        return dx

class Flatten:
    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        """
        This layer flattens the input tensor into a 2D array.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Flattened tensor of shape (B, C * H * W, 1)
        """
        B, _, _, _ = x.shape
        self.x_shape = x.shape
        return x.reshape(B, -1, 1)  # Reshape to (B, C * H * W, 1)

    def backward(self, grad):
        """
        This method reshapes the gradient back to the original input shape.
        :param grad: Gradient tensor of shape (B, C * H * W, 1)
        :return: Gradient reshaped to the original input shape (B, C, H, W)
        """
        return grad.reshape(self.x_shape)

class FullyConnected:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights and biases
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.random.randn(out_features, 1)

        # Buffer for gradients
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        Forward pass through the fully connected layer.
        :param x: Input tensor of shape (B, in_features, 1)
        :return: Output tensor of shape (B, out_features, 1)
        """
        self.x = x
        # Remove the last dimension for matmul, then add it back
        out = np.matmul(x[:, :, 0], self.W.T) + self.b.T  # (B, out_features)
        return out[:, :, np.newaxis]  # (B, out_features, 1)

    def backward(self, grad):
        """
        Backward pass through the fully connected layer.
        :param grad: Gradient tensor of shape (B, out_features, 1)
        :return: Gradient tensor of shape (B, in_features, 1)
        """
        B = grad.shape[0]
        self.dW += np.sum(np.matmul(grad, self.x.transpose(0, 2, 1)), axis=0)  # Sum over batch dimension
        self.db += np.sum(grad, axis=0)  # Sum over batch dimension

        dx = np.matmul(grad[:, :, 0], self.W)  # (B, I)
        dx = dx[:, :, np.newaxis]  # (B, I, 1)
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
        """
        logits: (B, num_classes, 1)
        labels: (B, num_classes, 1) for one-hot
        Returns: scalar loss (average over batch)
        """
        # Remove last dimension if present
        if logits.ndim == 3 and logits.shape[2] == 1:
            logits = logits[:, :, 0]
        if labels.ndim == 3 and labels.shape[2] == 1:
            labels = labels[:, :, 0]

        self.logits = logits
        self.labels = labels

        # Compute softmax probabilities for each sample in batch
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)  # (B, num_classes)

        # Compute cross-entropy loss for each sample, then average
        log_probs = -np.log(self.probs + 1e-10)
        loss = np.sum(labels * log_probs, axis=1)  # (B,)
        return np.mean(loss)  # scalar

    def backward(self):
        '''
        Compute the gradient of the loss with respect to the logits.
        :return: Gradient of the loss with respect to the logits, shape (B, num_classes, 1)
        '''
        # Don't divide by B here - let the layer updates handle the averaging
        grad = self.probs - self.labels  # (B, num_classes)
        return grad[:, :, np.newaxis]  # (B, num_classes, 1)