
import numpy as np

class ConvNet:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.layers = []

        # Optional configs
        self.batch_size = self.config.get('batch_size', 1)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.epochs = self.config.get('epochs', 10)

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

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
        return np.argmax(out)

    def evaluate(self, X, y):
        """Evaluate accuracy and loss on a dataset."""
        correct = 0
        total_loss = 0

        for x_i, y_i in zip(X, y):
            out = self.forward(x_i)
            pred = np.argmax(out)
            correct += int(pred == y_i)

            # Cross-entropy loss
            loss = -np.log(out[y_i] + 1e-10)
            total_loss += loss

        accuracy = correct / len(X)
        avg_loss = total_loss / len(X)
        return accuracy, avg_loss

    def SGD(self, X_train, y_train, X_val=None, y_val=None):
        for epoch in range(self.epochs):

            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            for i in range(0, len(X_train), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = [X_train[j] for j in batch_indices]
                y_batch = [y_train[j] for j in batch_indices]

                for x_i, y_i in zip(X_batch, y_batch):
                    self.forward(x_i)

                    self.backward()
                self.update(self.batch_size)

            # Validation evaluation
            if X_val is not None and y_val is not None:
                val_acc, val_loss = self.evaluate(X_val, y_val)
                print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
