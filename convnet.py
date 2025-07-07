import numpy as np
from convnet_layers import CrossEntropyLoss
from func_util import Softmax

class ConvNetConfig:
    """
    Configuration class for ConvNet architecture and training parameters.
    """
    def __init__(
        self,
        batch_size=1,
        learning_rate=0.01,
        epochs=10
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __str__(self):
        return (
            f"ConvNetConfig(batch_size={self.batch_size}, "
            f"learning_rate={self.learning_rate}, epochs={self.epochs})"
        )

class ConvNet:
    def __init__(self, config=ConvNetConfig()):
        self.config = config if config else {}
        self.layers = []

        # Optional configs
        self.batch_size = self.config.batch_size
        self.learning_rate = self.config.learning_rate
        self.epochs = self.config.epochs

        # Use cross entropy loss by default
        self.loss_fn = CrossEntropyLoss()

    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, batch_size):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(self.learning_rate, batch_size)

    def zero_grad(self):
        """Reset gradients for all layers."""
        for layer in self.layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()

    def predict(self, x):
        """Returns the predicted class for a single input sample."""
        x_batch = np.expand_dims(x, axis=0)  # Add batch dimension
        out = self.forward(x_batch)

        # Apply softmax to get probabilities
        probs = Softmax.fn(out[0])  # Get the first (and only) sample's output
        return np.argmax(probs)

    def evaluate(self, validation_data):
        """
        Evaluate the model on the validation dataset.
        returns the accuracy as a float.
        """
        total = len(validation_data)
        correct = 0

        # Process in batches
        batch_size = self.batch_size
        for i in range(0, total, batch_size):
            batch = validation_data[i:i + batch_size]
            x_batch = np.stack([x for x, _ in batch], axis=0)
            y_batch = [y for _, y in batch]

            out = self.forward(x_batch)

            # Apply softmax to get probabilities
            for y_true, y_pred in zip(y_batch, out):
                probs = Softmax.fn(y_pred)
                if np.argmax(probs) == y_true:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def SGD(self, train_data, validation_data=None):
        """
        Stochastic Gradient Descent optimization.
        """
        num_samples = len(train_data)
        loss_fn = self.loss_fn

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)
            for batch_start in range(0, num_samples, self.batch_size):
                batch = train_data[batch_start:batch_start + self.batch_size]

                # Prepare the batch data
                x_batch = np.stack([x for x, _ in batch], axis=0)
                y_batch = np.stack([y for _, y in batch], axis=0)

                # Reset gradients for the batch
                self.zero_grad()

                # Forward pass - batched version
                out = self.forward(x_batch)

                # Compute loss and gradients - batched version
                loss = loss_fn.forward(out, y_batch)
                grad = loss_fn.backward()

                # Backward pass - batched version
                self.backward(grad)

                # Update weights
                self.update(len(batch))

            if validation_data:
                acc = self.evaluate(validation_data)
                print(f"Epoch {epoch + 1}/{self.epochs}, Validation Accuracy: {acc * 100:.2f}%")
