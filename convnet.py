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

class ConvNet:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.layers = []

        # Optional configs
        self.batch_size = self.config.get('batch_size', 1)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.epochs = self.config.get('epochs', 10)

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

    def predict(self, x):
        """Returns the predicted class for a single input sample."""
        out = self.forward(x)

        # Apply softmax to get probabilities
        probs = Softmax.fn(out)
        return np.argmax(probs)

    def evaluate(self, validation_data):
        """
        Evaluate the model on the validation dataset.
        returns the accuracy as a float.
        """
        results = [(self.predict(x), y) for x, y in validation_data]
        return sum(int(y_pred == y_true) for y_pred, y_true in results) / len(results)

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

                for x, y in batch:
                    # Forward pass
                    out = self.forward(x)

                    # Compute loss and gradients
                    loss = loss_fn.forward(out, y)
                    grad = loss_fn.backward(out, y)

                    # Backward pass
                    self.backward(grad)

                # Update weights
                self.update(self.batch_size)

            if validation_data:
                self.evaluate(validation_data)
                print(f"Epoch {epoch + 1}/{self.epochs}, Validation Accuracy: {self.evaluate(validation_data):.4f}")
