import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize weights and biases
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))

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

        # Step 1: Unroll image patches into columns
        self.x_cols = self._im2col(x, kernel_size)  # shape: (B, H_out*W_out, C*K*K)

        # Step 2: Flatten filters to matrix: (C_out, C*K*K)
        W_col = self.W.reshape(self.out_channels, -1)  # (C_out, C*K*K)

        # Step 3: Matrix multiply and add bias
        out = self.x_cols @ W_col.T + self.b.T  # (B, H_out*W_out, C_out)

        # Step 4: Reshape to (B, C_out, H_out, W_out)
        out = out.transpose(0, 2, 1).reshape(B, self.out_channels, H_out, W_out)

        return out

    def backward(self, grad):
        """
        grad shape: (B, C_out, H_out, W_out)
        dx shape: (B, C_in, H_in, W_in)
        """
        B, _, H_in, W_in = self.x.shape
        _, _, H_out, W_out = grad.shape
        C_out, C_in, K, _ = self.W.shape

        grad_flat = grad.reshape(B, C_out, -1).transpose(0, 2, 1)  # (B, H_out*W_out, C_out)

        # grad_flat: (B, H_out*W_out, C_out), x_cols: (B, H_out*W_out, C_in*K*K)
        # dW_reshaped: (C_in*K*K, C_out)
        # This performs: sum over batch and positions (B, H_out*W_out)
        dW_reshaped = np.einsum('bic,bij->jc', grad_flat, self.x_cols)  # (C_in*K*K, C_out)
        self.dW += dW_reshaped.T.reshape(C_out, C_in, K, K)  # Reshape back to filter shape

        # db computation is already vectorized
        self.db += np.sum(grad_flat, axis=(0, 1)).reshape(C_out, 1)

        # Vectorized dx computation
        W_col = self.W.reshape(C_out, -1)  # (C_out, C_in*K*K)
        dx_cols = np.einsum('bic,cj->bij', grad_flat, W_col)  # (B, H_out*W_out, C_in*K*K)

        dx = self._col2im(dx_cols)
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

    def _im2col(self, x, K):
        B, C, H, W = x.shape
        H_out = H - K + 1
        W_out = W - K + 1

        # Pre-allocate the output array
        cols = np.zeros((B, H_out * W_out, C * K * K))

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i:i+K, j:j+K]  # (B, C, K, K)
                cols[:, i * W_out + j, :] = patch.reshape(B, -1)

        return cols

    def _col2im(self, dx_cols):
        B, C_in, H_in, W_in = self.x.shape
        K = self.kernel_size
        H_out = H_in - K + 1
        W_out = W_in - K + 1
        dx = np.zeros((B, C_in, H_in, W_in))

        for b in range(B):
            col_idx = 0
            for i in range(H_out):
                for j in range(W_out):
                    patch = dx_cols[b, col_idx].reshape(C_in, K, K)
                    dx[b, :, i:i+K, j:j+K] += patch
                    col_idx += 1

        return dx

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
        self.input = None
        self.max_indices = None

    def forward(self, x):
        """
        Ultra-fast forward pass using stride_tricks.
        :param x: Input tensor of shape (B, C, H_in, W_in)
        :return: Output tensor of shape (B, C, H_out, W_out)
        """
        self.input = x
        B, C, H_in, W_in = x.shape
        H_out = (H_in - self.pool_size) // self.stride + 1
        W_out = (W_in - self.pool_size) // self.stride + 1

        try:
            # Use stride_tricks for maximum performance
            from numpy.lib.stride_tricks import sliding_window_view

            # Create sliding windows
            windows = sliding_window_view(x, (self.pool_size, self.pool_size), axis=(2, 3))
            # Shape: (B, C, H_out, W_out, pool_size, pool_size)

            # Take every stride-th window
            windows = windows[:, :, ::self.stride, ::self.stride]

            # Flatten pool dimensions and find max
            windows_flat = windows.reshape(B, C, H_out, W_out, -1)
            max_vals = np.max(windows_flat, axis=-1)
            max_indices = np.argmax(windows_flat, axis=-1)

            # Store for backward pass
            self.max_indices = max_indices
            self.windows_shape = (H_out, W_out)

            return max_vals

        except ImportError:
            # Fallback to im2col method
            return self._forward_im2col(x, H_out, W_out)

    def backward(self, grad):
        """
        Optimized backward pass.
        :param grad: Gradient tensor of shape (B, C, H_out, W_out)
        :return: Gradient tensor of shape (B, C, H_in, W_in)
        """
        B, C, H_in, W_in = self.input.shape
        _, _, H_out, W_out = grad.shape

        dx = np.zeros_like(self.input)

        # Vectorized backward pass
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * self.stride
                w_start = w * self.stride

                # Get max indices for this output position
                max_idx = self.max_indices[:, :, h, w]  # (B, C)

                # Convert to 2D coordinates
                max_h = max_idx // self.pool_size
                max_w = max_idx % self.pool_size

                # Vectorized gradient assignment
                batch_idx = np.arange(B)[:, None]  # (B, 1)
                channel_idx = np.arange(C)[None, :]  # (1, C)

                h_coords = h_start + max_h
                w_coords = w_start + max_w

                dx[batch_idx, channel_idx, h_coords, w_coords] += grad[:, :, h, w]

        return dx

    def _forward_im2col(self, x, H_out, W_out):
        """Fallback im2col implementation"""
        B, C, H_in, W_in = x.shape

        # Extract windows more efficiently
        windows = np.zeros((B, C, H_out, W_out, self.pool_size * self.pool_size))

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * self.stride
                w_start = w * self.stride

                window = x[:, :, h_start:h_start + self.pool_size, w_start:w_start + self.pool_size]
                windows[:, :, h, w, :] = window.reshape(B, C, -1)

        # Find max values and indices
        max_vals = np.max(windows, axis=-1)
        max_indices = np.argmax(windows, axis=-1)

        self.max_indices = max_indices
        return max_vals

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